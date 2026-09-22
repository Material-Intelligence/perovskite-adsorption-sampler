"""Resumable single-molecule adsorption workflow (MLP screening, then DFT).

This is the core module of the recipes layer. It treats one molecule on one slab
as a single stateful task: every parameter and every intermediate result lives in
the task directory, so an interrupted run picks up where it stopped.

Public functions:
  - :func:`run_adsorption_task` runs the full workflow for one molecule;
  - :func:`run_dft_stage` runs the VASP and Bader stage on its own.

Examples:
    >>> from perovml.recipes.adsorption import run_adsorption_task  # doctest: +SKIP
    >>> result = run_adsorption_task(  # doctest: +SKIP
    ...     task_dir="outputs/mol_001",
    ...     slab=slab,
    ...     adsorbate=adsorbate,
    ...     calc=calculator,
    ...     fmax=0.03,
    ...     steps=150,
    ... )
"""

from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.io import read, write
from ase.optimize import LBFGS
from fairchem.core.components.calculate.recipes.adsorbml import detect_anomaly
from fairchem.data.oc.core.adsorbate import Adsorbate
from fairchem.data.oc.core.slab import Slab

from perovml.core.recipes import perov_adslab_generator, relax_job, select_best_by_priority
from perovml.utils.structure import build_oc_slab_from_atoms, ensure_surface_tags_and_constraints

logger = logging.getLogger(__name__)

__all__ = [
    "aggregate_and_rank_results",
    "generate_config_name",
    "generate_dual_sampling_configs",
    "relax_single_config",
    "run_adsorption_task",
    "run_dft_stage",
]

ORGANIC_ELEMENTS = frozenset({"C", "N", "H", "O", "F", "Cl", "Br", "S"})
"""Elements treated as belonging to the adsorbate rather than to the slab."""

TAG_BULK = 0
"""FAIRChem tag for a subsurface (bulk-like) atom."""
TAG_SURFACE = 1
"""FAIRChem tag for a surface atom."""
TAG_ADSORBATE = 2
"""FAIRChem tag for an adsorbate atom."""


# =============================================================================
# Small I/O helpers
# =============================================================================


def _load_json(path: Path) -> dict:
    """Read a JSON file, returning an empty dict only when there is no file.

    A task directory that has no ``status.json`` yet is an ordinary fresh start. A
    ``status.json`` that cannot be parsed is not: treating a truncated file as "nothing
    has been done" silently discards the record of what was, and the run would carry on
    from a state that never existed. So a corrupt file is an error the user has to look at.

    Args:
        path: File to read.

    Returns:
        The decoded object, or ``{}`` when the file does not exist.

    Raises:
        RuntimeError: If the file exists but cannot be read or decoded.
    """
    if not path.exists():
        return {}
    try:
        with open(path) as file:
            return json.load(file)
    except (OSError, ValueError) as exc:
        raise RuntimeError(
            f"{path} exists but could not be read as JSON ({exc}). Inspect it, then move it "
            "aside to restart this step from scratch."
        ) from exc


def _save_json(path: Path, data: dict) -> None:
    """Write a dict as indented JSON, creating parent directories as needed.

    Args:
        path: File to write.
        data: Object to serialise. NumPy scalars are converted automatically.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file:
        json.dump(data, file, indent=2, default=_json_default)


def _json_default(obj: Any) -> Any:
    """Make NumPy scalars JSON serialisable.

    Args:
        obj: Object that :mod:`json` could not serialise.

    Returns:
        A plain Python equivalent.

    Raises:
        TypeError: If the object is not a NumPy scalar either.
    """
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    raise TypeError(f"Object of type {obj.__class__.__name__} is not JSON serializable")


def _update_status(task_dir: Path, key: str, value: Any) -> dict:
    """Set one field in the task's ``status.json`` and write it back.

    Args:
        task_dir: Task directory.
        key: Field to set.
        value: New value.

    Returns:
        The updated status dict.
    """
    status = _load_json(task_dir / "status.json")
    status[key] = value
    _save_json(task_dir / "status.json", status)
    return status


def _default_seed(task_dir: Path) -> int:
    """Derive this task's sampling seed from its directory name.

    Deriving it rather than drawing it means every molecule in a campaign is reproducible
    and independent of the others, and that re-running one task reproduces its own
    candidate set rather than someone else's.

    Args:
        task_dir: The task directory.

    Returns:
        A seed in ``[0, 2**63)``.
    """
    digest = hashlib.blake2b(task_dir.name.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (2**63)


def _resolve_sampling_seed(task_dir: Path, config_path: Path, cfg: dict, already_generated: bool) -> int:
    """Return the seed to sample with, refusing to invent one for an existing task.

    Generating the candidates for the first time may pick a seed and record it. Generating
    them *again* -- because ``configs.traj`` was lost, say on a scratch filesystem -- may
    not: without the original seed the new candidates are a different set from the one the
    results already on disk belong to, and the task would silently become a mixture of two
    experiments.

    Args:
        task_dir: The task directory.
        config_path: Its ``config.json``, updated when a seed is assigned.
        cfg: The task configuration, updated in place.
        already_generated: Whether this task has generated its candidates before.

    Returns:
        The seed to pass to the samplers.

    Raises:
        RuntimeError: If the candidates have to be regenerated and no seed was recorded.
    """
    if cfg.get("seed") is not None:
        return int(cfg["seed"])
    if already_generated:
        raise RuntimeError(
            f"the candidates of {task_dir} have to be regenerated, but {config_path} records no "
            "seed, so the original set cannot be reproduced and the results already on disk would "
            "no longer belong to the same candidates. Pass the seed this task was started with as "
            "seed=..., or move the task directory aside and run it again."
        )
    cfg["seed"] = _default_seed(task_dir)
    _save_json(config_path, cfg)
    return int(cfg["seed"])


def _singlepoint_energy(atoms: Atoms, calc) -> float:
    """Evaluate the energy of a structure without relaxing it.

    Args:
        atoms: Structure to evaluate. It is copied, never modified.
        calc: ASE calculator.

    Returns:
        The potential energy, in eV.
    """
    atoms = atoms.copy()
    atoms.calc = calc
    energy = atoms.get_potential_energy()
    atoms.calc = None
    return float(energy)


def _write_vasp_sorted(path: Path, atoms: Atoms) -> None:
    """Write a POSCAR with the elements grouped, leaving the structure untouched.

    Args:
        path: File to write.
        atoms: Structure to write.
    """
    write(path, atoms, format="vasp", direct=True, sort=True, vasp5=True)


def _write_vasp_sorted_with_tags(path: Path, atoms: Atoms) -> None:
    """Write a POSCAR plus the side-car file that carries its tags.

    A POSCAR cannot hold ASE tags, and reading one back gives every atom tag 0. Anything
    downstream that has to know which atoms are the adsorbate -- the Bader charge transfer,
    above all -- reads the side-car instead, so the two are always written together and in
    the same order.

    Args:
        path: POSCAR to write. The tags go next to it as ``<stem>_tags.json``.
        atoms: Structure to write, carrying the tags.
    """
    # ase.io.vasp.write_vasp(sort=True) reorders with exactly this call -- alphabetically
    # by chemical symbol -- so repeating it verbatim is what keeps tags_sorted[i] pointing
    # at the POSCAR's atom i.
    order = np.argsort(atoms.symbols)
    tags_sorted = np.array(atoms.get_tags())[order].tolist()

    write(path, atoms, format="vasp", direct=True, sort=True, vasp5=True)
    _save_json(path.parent / f"{path.stem}_tags.json", {"tags": tags_sorted})


def _load_tags(path: Path) -> list[int]:
    """Read the tags written beside a POSCAR by :func:`_write_vasp_sorted_with_tags`.

    Args:
        path: The POSCAR, not the side-car.

    Returns:
        The tags, or an empty list when no side-car was written.
    """
    data = _load_json(path.parent / f"{path.stem}_tags.json")
    return [int(tag) for tag in data.get("tags", [])]


def _fixed_indices(atoms: Atoms) -> set[int]:
    """Collect the indices frozen by the structure's ``FixAtoms`` constraints.

    Args:
        atoms: Structure to inspect.

    Returns:
        The frozen atom indices, empty when nothing is constrained.
    """
    fixed: set[int] = set()
    for constraint in atoms.constraints:
        if isinstance(constraint, FixAtoms):
            # FixAtoms.index may be a NumPy array.
            for index in getattr(constraint, "index", []):
                fixed.add(int(index))
    return fixed


# =============================================================================
# Candidate generation and relaxation
# =============================================================================


def generate_config_name(source: str, index: int) -> str:
    """Build a candidate name such as ``uniform_000`` or ``heuristic_012``.

    Args:
        source: Sampler that produced the candidate.
        index: Index within that sampler's output.

    Returns:
        The candidate name.
    """
    return f"{source}_{index:03d}"


def _save_config_with_tags(config_dir: Path, name: str, atoms: Atoms) -> None:
    """Write a candidate as a POSCAR plus a side-car file holding its tags.

    The VASP format cannot carry ASE tags, so they are stored separately. To keep
    the two consistent, the atoms are sorted by atomic number in memory first and
    the tags are permuted the same way, which also groups the elements the way
    VASP and most viewers expect. The file is then written with ``sort=False``,
    because the sorting has already happened.

    Args:
        config_dir: Directory to write into.
        name: Candidate name, used as the file stem.
        atoms: Candidate structure.
    """
    atoms_sorted = atoms
    tags = atoms.get_tags()

    if len(tags) == len(atoms_sorted):
        numbers = np.array(atoms_sorted.get_atomic_numbers())
        order = np.argsort(numbers, kind="mergesort")  # stable, so elements stay blocked
        atoms_sorted = atoms_sorted[order]
        tags_sorted = np.array(tags)[order].tolist()
    else:
        # Should not happen; re-infer the tags before sorting rather than guessing.
        _infer_and_set_tags(atoms_sorted)
        numbers = np.array(atoms_sorted.get_atomic_numbers())
        order = np.argsort(numbers, kind="mergesort")
        atoms_sorted = atoms_sorted[order]
        tags_sorted = atoms_sorted.get_tags().tolist()

    write(
        config_dir / f"{name}.vasp",
        atoms_sorted,
        format="vasp",
        direct=True,
        sort=False,
        vasp5=True,
    )
    _save_json(config_dir / f"{name}_tags.json", {"tags": tags_sorted})


def _save_configs_archive(mlp_dir: Path, configs: list[dict[str, Any]]) -> None:
    """Write every candidate to a single ASE trajectory file.

    This archive is the authoritative record used for resuming and for anomaly
    detection: it preserves the atom order, the tags and the constraints exactly,
    so that the init, final and slab structures stay index-consistent.

    Args:
        mlp_dir: The task's ``mlp`` directory.
        configs: Candidates as produced by :func:`generate_dual_sampling_configs`.
    """
    archive_path = mlp_dir / "configs.traj"
    atoms_list: list[Atoms] = []
    for config in configs:
        atoms = config["atoms"].copy()
        info = atoms.info or {}
        info["config_name"] = config["name"]
        info["source"] = config["source"]
        info["source_index"] = config["index"]
        atoms.info = info
        atoms_list.append(atoms)
    if atoms_list:
        write(archive_path, atoms_list, format="traj")


def _load_configs_archive(mlp_dir: Path) -> dict[str, Atoms]:
    """Load the candidate archive and index it by candidate name.

    Args:
        mlp_dir: The task's ``mlp`` directory.

    Returns:
        A name-to-structure mapping, empty when the archive is missing or unreadable.
        The caller decides whether to regenerate the candidates in that case.
    """
    archive_path = mlp_dir / "configs.traj"
    if not archive_path.exists():
        return {}
    try:
        configs = read(archive_path, ":")
    except (OSError, ValueError, IndexError):
        logger.warning("could not read %s, treating the archive as missing", archive_path, exc_info=True)
        return {}
    name_to_atoms: dict[str, Atoms] = {}
    for atoms in configs:
        name = atoms.info.get("config_name") if atoms.info else None
        if isinstance(name, str) and name:
            name_to_atoms[name] = atoms
    return name_to_atoms


def _load_config_with_tags(config_dir: Path, name: str) -> Atoms:
    """Read a candidate POSCAR and restore its tags.

    The side-car tags are cross-checked against the constraints; if the two
    disagree the tags are re-inferred from the structure instead of being trusted.

    Args:
        config_dir: Directory holding the candidate files.
        name: Candidate name, used as the file stem.

    Returns:
        The candidate structure, tagged.
    """
    atoms = read(config_dir / f"{name}.vasp")
    tags_file = config_dir / f"{name}_tags.json"

    if not tags_file.exists():
        # Older runs did not write the side-car file.
        logger.warning("no tags file for %s, inferring the tags from the structure", name)
        _infer_and_set_tags(atoms)
        return atoms

    tags = _load_json(tags_file).get("tags", [])
    if len(tags) != len(atoms):
        logger.warning("tags length mismatch for %s, inferring the tags from the structure", name)
        _infer_and_set_tags(atoms)
        return atoms

    fixed_indices = _fixed_indices(atoms)
    if not fixed_indices:
        # No constraint information, so the stored tags are all there is to go on.
        atoms.set_tags(tags)
        return atoms

    # With constraints available, an unconstrained organic atom must be an adsorbate atom.
    symbols = atoms.get_chemical_symbols()
    inferred_ads_indices = {
        index for index in range(len(atoms)) if index not in fixed_indices and symbols[index] in ORGANIC_ELEMENTS
    }
    tag_ads_indices = {index for index, tag in enumerate(tags) if tag == TAG_ADSORBATE}
    # A clear disagreement means the tags no longer line up with the structure.
    if inferred_ads_indices and not tag_ads_indices.issubset(inferred_ads_indices):
        logger.warning("tags inconsistent with the constraints for %s, re-inferring them from the structure", name)
        _infer_and_set_tags(atoms)
    else:
        atoms.set_tags(tags)

    return atoms


def _infer_and_set_tags(atoms: Atoms) -> None:
    """Infer the FAIRChem tags (0 bulk, 1 surface, 2 adsorbate) and set them.

    The selective-dynamics constraints are used first, because they are reliable
    for the systems this package targets: in a FAPbI3 slab the interior and the FA
    molecules are all fixed, only the top Pb/I layer and the adsorbate are free,
    so "unconstrained" plus "organic element" identifies the adsorbate exactly.

    Without constraints the function falls back to a simple geometric heuristic
    based on the z coordinate.

    Args:
        atoms: Structure to tag. Modified in place.
    """
    positions = atoms.get_positions()
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(atoms)

    # 1) Constraint-based inference.
    fixed_indices = _fixed_indices(atoms)
    if fixed_indices:
        tags = np.zeros(n_atoms, dtype=int)
        for index, symbol in enumerate(symbols):
            if index in fixed_indices:
                tags[index] = TAG_BULK  # the slab interior, plus the FA molecules
            elif symbol in ORGANIC_ELEMENTS:
                tags[index] = TAG_ADSORBATE
            else:
                tags[index] = TAG_SURFACE  # the free top Pb/I layer
        atoms.set_tags(tags)
        return

    # 2) Fallback: z coordinate and element only. More general, less precise.
    z_coords = positions[:, 2]
    z_max = z_coords.max()
    z_min = z_coords.min()
    z_range = z_max - z_min if z_max > z_min else 1.0

    # The molecule normally sits above the surface: look in the top 30% of the slab.
    z_threshold = z_max - 0.3 * z_range

    tags = np.zeros(n_atoms, dtype=int)

    for index, (symbol, z) in enumerate(zip(symbols, z_coords)):
        if symbol in ORGANIC_ELEMENTS and z > z_threshold:
            tags[index] = TAG_ADSORBATE

    # Split the remaining atoms into surface and bulk at their median z.
    non_mol_indices = [index for index in range(n_atoms) if tags[index] != TAG_ADSORBATE]
    if non_mol_indices:
        z_median = np.median(z_coords[non_mol_indices])
        for index in non_mol_indices:
            tags[index] = TAG_SURFACE if z_coords[index] > z_median else TAG_BULK

    atoms.set_tags(tags)


def generate_dual_sampling_configs(
    slab: Slab,
    adsorbates_kwargs: list[dict[str, Any]],
    pb_num_uniform: int = 30,
    pb_num_heuristic: int = 12,
    pb_cone_deg: float = 25.0,
    rng: int | None = None,
) -> list[dict[str, Any]]:
    """Generate the candidates of both Pb samplers, without relaxing them.

    Args:
        slab: Slab to place the adsorbate on.
        adsorbates_kwargs: Adsorbate specs, see
            :func:`perovml.core.recipes.perov_adslab_generator`.
        pb_num_uniform: Number of uniformly sampled orientations.
        pb_num_heuristic: Number of cone-sampled orientations.
        pb_cone_deg: Cone half-angle in degrees for the heuristic sampler.
        rng: Seed for the uniform sampler. Pass the task's recorded seed, so that the
            same candidate set comes back if it ever has to be regenerated. The
            heuristic sampler is deterministic and needs none.

    Returns:
        One dict per candidate, with the keys ``atoms``, ``name`` (e.g.
        ``uniform_000``), ``source`` (``uniform`` or ``heuristic``) and ``index``.
    """
    configs: list[dict[str, Any]] = []

    uniform_atoms = perov_adslab_generator(
        slab,
        adsorbates_kwargs=adsorbates_kwargs,
        generator="Pb_uniform_sample",
        pb_num_orientations=pb_num_uniform,
        rng=rng,
    )
    for index, atoms in enumerate(uniform_atoms):
        configs.append(
            {
                "atoms": atoms,
                "name": generate_config_name("uniform", index),
                "source": "uniform",
                "index": index,
            }
        )

    heuristic_atoms = perov_adslab_generator(
        slab,
        adsorbates_kwargs=adsorbates_kwargs,
        generator="Pb_heuristic_sample",
        pb_num_orientations=pb_num_heuristic,
        pb_cone_deg=pb_cone_deg,
    )
    for index, atoms in enumerate(heuristic_atoms):
        configs.append(
            {
                "atoms": atoms,
                "name": generate_config_name("heuristic", index),
                "source": "heuristic",
                "index": index,
            }
        )

    return configs


def relax_single_config(
    config: dict[str, Any],
    ml_relax_job,
    relaxed_slab_atoms: Atoms | None = None,
    relaxed_slab_energy: float | None = None,
    relaxed_gas_adsorbate_energy: float | None = None,
    disable_anomaly_detection: bool = False,
) -> dict[str, Any]:
    """Relax one candidate and attach its adsorption energy and anomalies.

    Anomaly detection follows the FAIRChem original: ``detect_anomaly`` compares
    the initial and the final adslab, and needs correct tags on the initial
    structure (0 bulk, 1 surface, 2 adsorbate).

    Args:
        config: One candidate from :func:`generate_dual_sampling_configs`.
        ml_relax_job: Callable taking one ``Atoms`` and returning a ``relax_job``-shaped dict.
        relaxed_slab_atoms: Relaxed bare slab. Accepted for interface symmetry;
            anomaly detection uses the slab part of the initial adslab instead.
        relaxed_slab_energy: Relaxed bare-slab energy, in eV.
        relaxed_gas_adsorbate_energy: Relaxed gas-phase adsorbate energy, in eV.
        disable_anomaly_detection: Skip anomaly detection and report none.

    Returns:
        The ``relax_job`` result, with ``adsorption_energy``, ``anomalies``,
        ``source`` and ``source_index`` added to ``results`` and the candidate name
        added as ``config_name``.
    """
    init_atoms = config["atoms"]
    result = ml_relax_job(init_atoms)
    final_atoms = result["atoms"]

    # Relaxation drops the tags, so copy them over from the initial structure.
    if len(final_atoms) == len(init_atoms):
        final_atoms.set_tags(init_atoms.get_tags())

    if disable_anomaly_detection:
        anomalies: list[str] = []
    else:
        try:
            tags = init_atoms.get_tags()
            n_mol_atoms = sum(1 for tag in tags if tag == TAG_ADSORBATE)

            if n_mol_atoms == 0:
                logger.warning(
                    "no adsorbate atoms (tag=2) in %s, the tags are probably missing; inferring them",
                    config.get("name"),
                )
                _infer_and_set_tags(init_atoms)

            # An earlier version compared against a separate bare-slab reference
            # (relaxed_slab_atoms). With precomputed slabs and resumed runs, small
            # differences between references marked every candidate as
            # surface_changed. Passing final_slab_atoms=None reproduces FAIRChem's
            # own default: use the slab part of the initial adslab as the reference.
            anomalies = detect_anomaly(init_atoms, final_atoms, final_slab_atoms=None)
        except Exception as exc:  # noqa: BLE001 - detection must never abort the run
            logger.warning("detect_anomaly failed for %s: %s", config.get("name"), exc)
            anomalies = ["anomaly_detection_failed"]

    e_ads = None
    if relaxed_slab_energy is not None and relaxed_gas_adsorbate_energy is not None:
        e_ads = float(result["results"]["energy"]) - relaxed_slab_energy - relaxed_gas_adsorbate_energy

    result["results"]["adsorption_energy"] = e_ads
    result["results"]["anomalies"] = anomalies
    result["results"]["source"] = config["source"]
    result["results"]["source_index"] = config["index"]
    result["config_name"] = config["name"]

    return result


def aggregate_and_rank_results(
    results: list[dict[str, Any]],
    relaxed_slab_energy: float | None = None,
    relaxed_gas_adsorbate_energy: float | None = None,
) -> dict[str, Any]:
    """Rank every relaxed candidate and count how many came from each sampler.

    Args:
        results: Relaxed candidates from :func:`relax_single_config`.
        relaxed_slab_energy: Relaxed bare-slab energy, in eV, for the report.
        relaxed_gas_adsorbate_energy: Relaxed gas-phase energy, in eV, for the report.

    Returns:
        A dict with the ranked ``adslabs``, the ``references`` and a
        ``sampling_stats`` breakdown per sampler.
    """
    sorted_results = select_best_by_priority(results, n=len(results))

    uniform_count = sum(1 for result in results if result["results"].get("source") == "uniform")
    heuristic_count = sum(1 for result in results if result["results"].get("source") == "heuristic")

    return {
        "adslabs": sorted_results,
        "references": {
            "slab_energy": relaxed_slab_energy,
            "gas_adsorbate_energy": relaxed_gas_adsorbate_energy,
        },
        "sampling_stats": {
            "uniform_count": uniform_count,
            "heuristic_count": heuristic_count,
            "total_count": len(results),
        },
    }


# =============================================================================
# The task function
# =============================================================================


def run_adsorption_task(
    task_dir: str | Path,
    slab: Slab | Atoms,
    adsorbate: Adsorbate,
    calc,
    optimizer_cls=LBFGS,
    # Calculation parameters: required on the first run, reloaded from config.json afterwards
    fmax: float | None = None,
    steps: int | None = None,
    pb_num_uniform: int | None = None,
    pb_num_heuristic: int | None = None,
    pb_cone: float | None = None,
    save_top: int | None = None,
    seed: int | None = None,
    # Precomputed references
    precomputed_slab_atoms: Atoms | None = None,
    precomputed_slab_energy: float | None = None,
    # Provenance
    slab_file: str | None = None,
    molecule_file: str | None = None,
    # DFT options
    run_dft: bool = False,
    dft_mode: str = "scf",
    vasp_command: str = "vasp_std",
    dft_incar_override: dict | None = None,
    disable_anomaly_detection: bool = False,
    dft_generate_only: bool = False,
    potcar_dir: str | None = None,
    slab_mlp_relaxed: bool | None = None,
    slab_dft_dir: str | None = None,
) -> dict[str, Any]:
    """Run the adsorption workflow for one molecule, resuming if possible.

    All state lives in ``task_dir``, so calling the function again on the same
    directory continues from the last finished step rather than starting over.
    On the first call the calculation parameters are written to ``config.json``;
    on later calls they are read back from it, and a parameter that disagrees with
    the stored value is reported and ignored.

    The directory layout is::

        task_dir/
        |-- config.json      # parameters, written once and reloaded on resume
        |-- status.json      # progress tracking
        |-- mlp/
        |   |-- molecule.vasp / .json
        |   |-- slab.vasp / .json
        |   |-- best.vasp
        |   |-- configs.traj / results.traj
        |   |-- all_results.json
        |   `-- results/
        |-- configs/
        |   |-- uniform_000.vasp
        |   `-- ...
        `-- dft/             # only when run_dft or dft_generate_only is set
            |-- molecule/
            |-- slab/
            `-- adslab/

    Args:
        task_dir: Task directory. Everything this task produces lives here.
        slab: Slab structure. Plain ``Atoms`` are wrapped with
            :func:`~perovml.utils.structure.build_oc_slab_from_atoms`.
        adsorbate: The adsorbate to place.
        calc: ASE calculator used for the MLP relaxations.
        optimizer_cls: ASE optimizer class.
        fmax: Force convergence threshold in eV/A. Defaults to 0.03 on a first run.
        steps: Maximum optimizer steps. Defaults to 150 on a first run.
        pb_num_uniform: Number of uniformly sampled orientations. Defaults to 30.
        pb_num_heuristic: Number of cone-sampled orientations. Defaults to 12.
        pb_cone: Cone half-angle in degrees. Defaults to 25.
        save_top: How many ranked structures to keep. Defaults to 5.
        seed: Seed for the uniform sampler, recorded in ``config.json``. Defaults to one
            derived from the task directory name, which makes each molecule reproducible
            and independent of the others.
        precomputed_slab_atoms: Pre-relaxed slab structure.
        precomputed_slab_energy: Pre-relaxed slab energy in eV; skips the slab relaxation.
        slab_file: Path the slab came from, recorded for provenance.
        molecule_file: Path the molecule came from, recorded for provenance.
        run_dft: Whether to run the DFT stage after the MLP screening.
        dft_mode: ``"scf"`` for single points or ``"opt"`` for relaxations.
        vasp_command: Command used to launch VASP.
        dft_incar_override: INCAR entries layered on top of the defaults.
        disable_anomaly_detection: Skip anomaly detection during the MLP stage.
        dft_generate_only: Write the VASP inputs but do not run VASP.
        potcar_dir: POTCAR directory; falls back to ``VASP_PP_PATH``.
        slab_mlp_relaxed: Treat the supplied slab as already relaxed and only
            evaluate its energy.
        slab_dft_dir: An existing DFT slab directory to symlink instead of
            recomputing the bare slab.

    Returns:
        A summary dict with the reference energies, the best candidate and its
        metrics, the candidate counts and the ``mlp_done`` / ``dft_done`` flags.
    """
    task_dir = Path(task_dir)
    task_dir.mkdir(parents=True, exist_ok=True)

    config_path = task_dir / "config.json"
    status_path = task_dir / "status.json"

    # =========================================================================
    # Parameters: save them on the first run, reload them when resuming
    # =========================================================================
    existing_config = _load_json(config_path)

    if existing_config:
        cfg = existing_config
        # Patch in options that did not exist when this task was first created.
        for key, value in {
            "dft_incar_override": dft_incar_override,
            "disable_anomaly_detection": disable_anomaly_detection,
            "dft_generate_only": dft_generate_only,
            "potcar_dir": potcar_dir,
            "slab_mlp_relaxed": slab_mlp_relaxed,
            "slab_dft_dir": slab_dft_dir,
        }.items():
            if key not in cfg and value is not None:
                cfg[key] = value
                _save_json(config_path, cfg)
        # Report parameter drift, but keep going with the stored values.
        for key, passed_val in {
            "fmax": fmax,
            "steps": steps,
            "pb_num_uniform": pb_num_uniform,
            "pb_num_heuristic": pb_num_heuristic,
        }.items():
            saved_val = cfg.get(key)
            if passed_val is not None and saved_val is not None and passed_val != saved_val:
                logger.warning(
                    "parameter %s differs: passed=%s, stored=%s; using the stored value",
                    key,
                    passed_val,
                    saved_val,
                )
    else:
        cfg = {
            "fmax": fmax if fmax is not None else 0.03,
            "steps": steps if steps is not None else 150,
            "pb_num_uniform": pb_num_uniform if pb_num_uniform is not None else 30,
            "pb_num_heuristic": pb_num_heuristic if pb_num_heuristic is not None else 12,
            "pb_cone": pb_cone if pb_cone is not None else 25.0,
            "save_top": save_top if save_top is not None else 5,
            "seed": seed if seed is not None else _default_seed(task_dir),
            "slab_file": slab_file,
            "molecule_file": molecule_file,
            "run_dft": run_dft,
            "dft_mode": dft_mode,
            "vasp_command": vasp_command,
            "dft_incar_override": dft_incar_override,
            "disable_anomaly_detection": disable_anomaly_detection,
            "dft_generate_only": dft_generate_only,
            "potcar_dir": potcar_dir,
            "slab_mlp_relaxed": slab_mlp_relaxed,
            "slab_dft_dir": slab_dft_dir,
        }
        _save_json(config_path, cfg)

    status = _load_json(status_path)

    def ml_relax_job(atoms: Atoms) -> dict[str, Any]:
        """Relax one structure with this task's calculator and settings."""
        return relax_job(atoms, calc=calc, optimizer_cls=optimizer_cls, fmax=cfg["fmax"], steps=cfg["steps"])

    # A slab read from a POSCAR carries no ASE tags, but the Slab constructor needs
    # surface tags and constraints; build_oc_slab_from_atoms adds both.
    slab_obj = build_oc_slab_from_atoms(slab, min_ab=8.0) if isinstance(slab, Atoms) else slab

    # =========================================================================
    # Step 1: relax the bare slab, or take the precomputed reference
    # =========================================================================
    mlp_dir = task_dir / "mlp"
    mlp_dir.mkdir(parents=True, exist_ok=True)
    slab_energy: float | None = None
    slab_relaxed: Atoms | None = None

    if precomputed_slab_energy is not None:
        slab_energy = float(precomputed_slab_energy)
        slab_relaxed = precomputed_slab_atoms if precomputed_slab_atoms is not None else slab_obj.atoms
        if slab_relaxed is not None:
            slab_relaxed = ensure_surface_tags_and_constraints(slab_relaxed)
        if not (mlp_dir / "slab.vasp").exists():
            _write_vasp_sorted(mlp_dir / "slab.vasp", slab_relaxed)
        _save_json(
            mlp_dir / "slab.json",
            {"energy": slab_energy, "fmax": None, "converged": True, "nsteps": 0},
        )
        _update_status(task_dir, "slab_optimized", True)
        _update_status(task_dir, "slab_energy", slab_energy)
    elif cfg.get("slab_mlp_relaxed"):
        slab_relaxed = slab_obj.atoms
        slab_energy = _singlepoint_energy(slab_relaxed, calc)
        _write_vasp_sorted(mlp_dir / "slab.vasp", slab_relaxed)
        _save_json(
            mlp_dir / "slab.json",
            {"energy": slab_energy, "fmax": None, "converged": True, "nsteps": 0},
        )
        _update_status(task_dir, "slab_optimized", True)
        _update_status(task_dir, "slab_energy", slab_energy)
    elif status.get("slab_optimized"):
        slab_data = _load_json(mlp_dir / "slab.json")
        if slab_data and "energy" in slab_data and (mlp_dir / "slab.vasp").exists():
            slab_energy = slab_data["energy"]
            slab_relaxed = ensure_surface_tags_and_constraints(read(mlp_dir / "slab.vasp"))
        else:
            # The stored slab is unusable; reset and relax it again.
            _update_status(task_dir, "slab_optimized", False)
            status["slab_optimized"] = False

    if not status.get("slab_optimized") and slab_energy is None:
        slab_result = ml_relax_job(slab_obj.atoms)
        slab_energy = float(slab_result["results"]["energy"])
        slab_relaxed = ensure_surface_tags_and_constraints(slab_result["atoms"])

        _write_vasp_sorted(mlp_dir / "slab.vasp", slab_relaxed)
        _save_json(
            mlp_dir / "slab.json",
            {
                "energy": slab_energy,
                "fmax": float(slab_result["results"].get("fmax", 0)),
                "converged": slab_result["results"].get("converged"),
                "nsteps": slab_result["results"].get("nsteps"),
            },
        )
        _update_status(task_dir, "slab_optimized", True)
        _update_status(task_dir, "slab_energy", slab_energy)

    # =========================================================================
    # Step 2: relax the gas-phase molecule
    # =========================================================================
    gas_energy: float | None = None

    if status.get("molecule_optimized"):
        gas_data = _load_json(mlp_dir / "molecule.json")
        if gas_data and "energy" in gas_data:
            gas_energy = gas_data["energy"]
        else:
            _update_status(task_dir, "molecule_optimized", False)
            status["molecule_optimized"] = False

    if gas_energy is None:
        gas_result = ml_relax_job(adsorbate.atoms)
        gas_energy = float(gas_result["results"]["energy"])

        _write_vasp_sorted(mlp_dir / "molecule.vasp", gas_result["atoms"])
        _save_json(
            mlp_dir / "molecule.json",
            {
                "energy": gas_energy,
                "fmax": float(gas_result["results"].get("fmax", 0)),
                "converged": gas_result["results"].get("converged"),
                "nsteps": gas_result["results"].get("nsteps"),
            },
        )
        _update_status(task_dir, "molecule_optimized", True)
        _update_status(task_dir, "molecule_energy", gas_energy)

    # =========================================================================
    # Step 3: generate the sampled candidates
    # =========================================================================
    configs_dir = task_dir / "configs"
    # slab_relaxed may have come back from a POSCAR, which carries no tags, so make
    # sure the tags and constraints are complete before sampling on it.
    sampling_slab = (
        Slab(slab_atoms=ensure_surface_tags_and_constraints(slab_relaxed)) if slab_relaxed is not None else slab_obj
    )

    # mlp/configs.traj is the authoritative candidate record: it keeps the atom
    # order, the tags and the constraints, and drives both resuming and anomaly
    # detection. The .vasp and *_tags.json files under configs/ are for viewing and
    # manual inspection only, and are never read back for the calculation.
    already_generated = bool(status.get("configs_generated"))
    if not already_generated or not (mlp_dir / "configs.traj").exists():
        seed = _resolve_sampling_seed(task_dir, config_path, cfg, already_generated)
        if already_generated:
            logger.warning(
                "configs.traj is missing in %s; regenerating the candidates from the recorded seed %s",
                task_dir,
                seed,
            )
        configs = generate_dual_sampling_configs(
            slab=sampling_slab,
            adsorbates_kwargs=[{"adsorbate": adsorbate}],
            pb_num_uniform=cfg["pb_num_uniform"],
            pb_num_heuristic=cfg["pb_num_heuristic"],
            pb_cone_deg=cfg["pb_cone"],
            rng=cfg["seed"],
        )

        config_names = []
        configs_dir.mkdir(parents=True, exist_ok=True)
        for config in configs:
            _save_config_with_tags(configs_dir, config["name"], config["atoms"])
            config_names.append(config["name"])

        _save_configs_archive(mlp_dir, configs)

        _update_status(task_dir, "configs_generated", True)
        _update_status(task_dir, "config_names", config_names)
        _update_status(task_dir, "configs_completed", [])

    status = _load_json(status_path)

    # =========================================================================
    # Step 4: relax every candidate, one resumable step at a time
    # =========================================================================
    completed = set(status.get("configs_completed", []))
    results_dir = mlp_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_traj = mlp_dir / "results.traj"

    # Load the authoritative archive; regenerate it once if it is missing or short.
    name_to_atoms = _load_configs_archive(mlp_dir)
    config_names = status.get("config_names", [])
    if not name_to_atoms or any(name not in name_to_atoms for name in config_names):
        # Regenerating is only safe because the recorded seed reproduces the same
        # candidates; _resolve_sampling_seed refuses when there is none to reuse.
        seed = _resolve_sampling_seed(task_dir, config_path, cfg, already_generated=True)
        logger.warning(
            "configs.traj is missing or incomplete in %s; regenerating the candidates from the recorded seed %s",
            task_dir,
            seed,
        )
        configs = generate_dual_sampling_configs(
            slab=sampling_slab,
            adsorbates_kwargs=[{"adsorbate": adsorbate}],
            pb_num_uniform=cfg["pb_num_uniform"],
            pb_num_heuristic=cfg["pb_num_heuristic"],
            pb_cone_deg=cfg["pb_cone"],
            rng=cfg["seed"],
        )
        config_names = []
        configs_dir.mkdir(parents=True, exist_ok=True)
        for config in configs:
            _save_config_with_tags(configs_dir, config["name"], config["atoms"])
            config_names.append(config["name"])
        _save_configs_archive(mlp_dir, configs)
        _update_status(task_dir, "configs_generated", True)
        _update_status(task_dir, "config_names", config_names)
        status = _load_json(status_path)
        completed = set(status.get("configs_completed", []))
        name_to_atoms = _load_configs_archive(mlp_dir)

    all_results: list[dict[str, Any]] = []

    for name in config_names:
        result_json = results_dir / f"{name}.json"
        result_vasp = results_dir / f"{name}.vasp"

        if name in completed and result_json.exists():
            result_data = _load_json(result_json)
            if result_data and "energy" in result_data:
                result = {
                    "config_name": name,
                    "results": {
                        "energy": result_data["energy"],
                        "adsorption_energy": result_data.get("adsorption_energy"),
                        "fmax": result_data.get("fmax"),
                        "converged": result_data.get("converged"),
                        "anomalies": result_data.get("anomalies", []),
                        "source": result_data.get("source"),
                        "source_index": result_data.get("source_index"),
                    },
                }
                if result_vasp.exists():
                    result["atoms"] = read(result_vasp)
                all_results.append(result)
                continue
            # The stored result is unusable; drop it and recompute.
            completed.discard(name)

        init_atoms = name_to_atoms.get(name)
        if init_atoms is None:
            logger.warning("candidate %s is not in configs.traj, skipping it", name)
            continue
        parts = name.rsplit("_", 1)
        source = parts[0] if len(parts) == 2 else "unknown"
        index = int(parts[1]) if len(parts) == 2 and parts[1].isdigit() else 0

        config = {"atoms": init_atoms, "name": name, "source": source, "index": index}

        result = relax_single_config(
            config,
            ml_relax_job,
            relaxed_slab_atoms=slab_relaxed,
            relaxed_slab_energy=slab_energy,
            relaxed_gas_adsorbate_energy=gas_energy,
            disable_anomaly_detection=cfg.get("disable_anomaly_detection", False),
        )

        _write_vasp_sorted(result_vasp, result["atoms"])
        _save_json(
            result_json,
            {
                "config_name": name,
                "energy": result["results"]["energy"],
                "adsorption_energy": result["results"].get("adsorption_energy"),
                "fmax": result["results"].get("fmax"),
                "converged": result["results"].get("converged"),
                "nsteps": result["results"].get("nsteps"),
                "anomalies": result["results"].get("anomalies", []),
                "source": result["results"].get("source"),
                "source_index": result["results"].get("source_index"),
            },
        )

        # Also append to one trajectory file, which is easier to inspect afterwards.
        atoms_for_traj = result["atoms"].copy()
        info = atoms_for_traj.info or {}
        info["config_name"] = name
        info["source"] = result["results"].get("source")
        info["source_index"] = result["results"].get("source_index")
        info["energy"] = result["results"]["energy"]
        info["adsorption_energy"] = result["results"].get("adsorption_energy")
        info["anomalies"] = result["results"].get("anomalies", [])
        atoms_for_traj.info = info
        write(results_traj, atoms_for_traj, format="traj", append=True)

        completed.add(name)
        _update_status(task_dir, "configs_completed", list(completed))

        all_results.append(result)

    # =========================================================================
    # Step 5: rank the candidates and record the best one
    # =========================================================================
    if not status.get("mlp_done"):
        aggregated = aggregate_and_rank_results(
            all_results,
            relaxed_slab_energy=slab_energy,
            relaxed_gas_adsorbate_energy=gas_energy,
        )

        total_configs = len(all_results)
        converged_configs = sum(1 for result in all_results if bool(result.get("results", {}).get("converged")))

        summary_data: dict[str, Any] = {
            "config": cfg,
            "slab_energy": slab_energy,
            "gas_energy": gas_energy,
            "sampling_stats": aggregated["sampling_stats"],
            "convergence_stats": {"total": total_configs, "converged": converged_configs},
            "results": [],
        }

        for rank, result in enumerate(aggregated["adslabs"], start=1):
            summary_data["results"].append(
                {
                    "rank": rank,
                    "config_name": result.get("config_name"),
                    "energy": result["results"]["energy"],
                    "adsorption_energy": result["results"].get("adsorption_energy"),
                    "converged": result["results"].get("converged"),
                    "fmax": result["results"].get("fmax"),
                    "nsteps": result["results"].get("nsteps"),
                    "anomalies": result["results"].get("anomalies", []),
                    "source": result["results"].get("source"),
                }
            )

        _save_json(mlp_dir / "all_results.json", summary_data)

        top_results = select_best_by_priority(all_results, n=cfg["save_top"])

        if top_results:
            best = top_results[0]
            if "atoms" in best:
                # With its tags: the DFT stage needs to know which atoms are the adsorbate,
                # and a POSCAR on its own cannot say.
                _write_vasp_sorted_with_tags(mlp_dir / "best.vasp", best["atoms"])

            _update_status(task_dir, "best_config", best.get("config_name"))
            _update_status(task_dir, "best_energy", best["results"]["energy"])
            _update_status(task_dir, "best_adsorption_energy", best["results"].get("adsorption_energy"))
            _update_status(task_dir, "best_anomalies", best["results"].get("anomalies", []))
            _update_status(task_dir, "best_converged", best["results"].get("converged"))
            _update_status(task_dir, "best_fmax", best["results"].get("fmax"))
            _update_status(task_dir, "best_nsteps", best["results"].get("nsteps"))

        _update_status(task_dir, "configs_total", total_configs)
        _update_status(task_dir, "configs_converged", converged_configs)

        _update_status(task_dir, "mlp_done", True)

    # =========================================================================
    # Step 6: the optional DFT stage
    # =========================================================================
    status = _load_json(status_path)

    if (cfg.get("run_dft") or cfg.get("dft_generate_only")) and not status.get("dft_done"):
        run_dft_stage(
            task_dir,
            cfg,
            skip_run=cfg.get("dft_generate_only", False) and not cfg.get("run_dft", False),
        )

    status = _load_json(status_path)

    return {
        "task_dir": str(task_dir),
        "config": cfg,
        "slab_energy": status.get("slab_energy"),
        "molecule_energy": status.get("molecule_energy"),
        "best_config": status.get("best_config"),
        "best_energy": status.get("best_energy"),
        "best_adsorption_energy": status.get("best_adsorption_energy"),
        "best_anomalies": status.get("best_anomalies", []),
        "best_converged": status.get("best_converged"),
        "best_fmax": status.get("best_fmax"),
        "best_nsteps": status.get("best_nsteps"),
        "configs_total": status.get("configs_total"),
        "configs_converged": status.get("configs_converged"),
        "mlp_done": status.get("mlp_done", False),
        "dft_done": status.get("dft_done", False),
    }


# =============================================================================
# The DFT stage
# =============================================================================


def run_dft_stage(task_dir: str | Path, cfg: dict, skip_run: bool = False) -> None:
    """Run the VASP and Bader stage for a task that finished MLP screening.

    Three calculations are prepared in turn -- the gas-phase molecule, the bare
    slab and the best adslab -- and the DFT adsorption energy is derived from
    them. When the adslab produced a CHGCAR, a Bader analysis follows and the
    charge transferred onto the adsorbate is recorded.

    Every step checks ``status.json`` first, so the stage is resumable, and it
    returns early rather than raising when a VASP run fails.

    Args:
        task_dir: Task directory, as used by :func:`run_adsorption_task`.
        cfg: The task configuration, normally loaded from ``config.json``.
        skip_run: Only write the VASP inputs; do not launch VASP or Bader.

    Raises:
        RuntimeError: If ``slab_dft_dir`` points at a directory that looks
            inaccessible, which would otherwise be mistaken for a finished run.
    """
    from perovml.dft import VASPRunner, calculate_charge_transfer, combine_aeccar, load_zvals, run_bader

    task_dir = Path(task_dir)
    status = _load_json(task_dir / "status.json")
    mlp_dir = task_dir / "mlp"
    dft_dir = task_dir / "dft"

    vasp_cmd = cfg.get("vasp_command", "vasp_std")
    vasp = VASPRunner(potcar_dir=cfg.get("potcar_dir"))
    dft_mode = cfg.get("dft_mode", "scf")
    base_incar_override = cfg.get("dft_incar_override") or cfg.get("vasp_incar_override") or {}
    dft_calc_type = "opt" if dft_mode == "opt" else "scf"

    def _prepare_incar_override(calc_type: str) -> dict[str, Any]:
        """Fill in NSW and IBRION for the calculation type, unless the user set them.

        Args:
            calc_type: ``"scf"`` or ``"opt"``.

        Returns:
            The INCAR override dict for this calculation.
        """
        incar = dict(base_incar_override)
        nsw = incar.get("NSW")
        ibrion = incar.get("IBRION")
        if calc_type == "opt":
            if nsw is None or (isinstance(nsw, str) and nsw.lower() == "auto"):
                incar["NSW"] = 500
            if ibrion is None or (isinstance(ibrion, str) and ibrion.lower() == "auto"):
                incar["IBRION"] = 2
        else:
            if nsw is None or (isinstance(nsw, str) and nsw.lower() == "auto"):
                incar["NSW"] = 0
            if ibrion is None or (isinstance(ibrion, str) and ibrion.lower() == "auto"):
                incar["IBRION"] = -1
        return incar

    def _generate_inputs(atoms: Atoms, outdir: Path, calc_type: str) -> None:
        """Write the VASP inputs for one structure.

        Args:
            atoms: Structure to write.
            outdir: Directory for the input files.
            calc_type: ``"scf"`` or ``"opt"``.
        """
        vasp.generate_inputs(
            atoms,
            outdir,
            calc_type=calc_type,
            incar_override=_prepare_incar_override(calc_type),
        )

    def _accept_energy(stage: str, parsed: dict[str, Any]) -> float | None:
        """Return the parsed energy only if it may be used, and record why when it may not.

        An energy from a run that exhausted NELM, or from one VASP abandoned, looks exactly
        like a good one: the number is there and the process exited 0. Everything after this
        point -- the adsorption energy, the ranking, the campaign report -- takes the number
        at face value, so this is where a bad one has to stop.

        Args:
            stage: ``"molecule"``, ``"slab"`` or ``"adslab"``, for the message.
            parsed: What :meth:`~perovml.dft.vasp.VASPRunner.parse_results` returned.

        Returns:
            The energy in eV, or None when the stage must not be marked done.
        """
        energy = parsed.get("energy")
        if energy is None or not parsed.get("converged"):
            reason = parsed.get("error") or ("no energy in the OUTCAR" if energy is None else "did not converge")
            logger.error("DFT %s is not usable: %s", stage, reason)
            _update_status(task_dir, f"dft_{stage}_error", reason)
            _update_status(task_dir, "dft_error", f"{stage}: {reason}")
            return None
        _update_status(task_dir, f"dft_{stage}_error", None)
        return float(energy)

    # ---- DFT: the gas-phase molecule -----------------------------------------
    if cfg.get("dft_molecule_energy") is not None:
        _update_status(task_dir, "dft_molecule_energy", cfg["dft_molecule_energy"])
        _update_status(task_dir, "dft_molecule_done", True)
    elif not status.get("dft_molecule_done"):
        mol_atoms = read(mlp_dir / "molecule.vasp")
        mol_dft_dir = dft_dir / "molecule"

        # The molecule is used as supplied or as the MLP left it, vacuum included;
        # it is deliberately not re-centred.
        _generate_inputs(mol_atoms, mol_dft_dir, dft_calc_type)
        if skip_run:
            _update_status(task_dir, "dft_inputs_molecule", True)
        else:
            result = vasp.run(mol_dft_dir, vasp_cmd=vasp_cmd)
            if not result["success"]:
                logger.error("DFT molecule calculation failed: %s", result.get("error"))
                return
            energy = _accept_energy("molecule", vasp.parse_results(mol_dft_dir))
            if energy is None:
                return
            _update_status(task_dir, "dft_molecule_energy", energy)
            _update_status(task_dir, "dft_molecule_done", True)

    # ---- DFT: the bare slab --------------------------------------------------
    status = _load_json(task_dir / "status.json")
    slab_external_dir = None
    slab_dft_dir_cfg = cfg.get("slab_dft_dir")
    if slab_dft_dir_cfg:
        candidate = Path(slab_dft_dir_cfg)
        if candidate.is_dir():
            slab_external_dir = candidate.resolve()

    if cfg.get("dft_slab_energy") is not None:
        _update_status(task_dir, "dft_slab_energy", cfg["dft_slab_energy"])
        _update_status(task_dir, "dft_slab_done", True)
    elif slab_external_dir is not None:
        slab_dft_dir = dft_dir / "slab"
        if slab_dft_dir.exists() or slab_dft_dir.is_symlink():
            if slab_dft_dir.is_symlink() or not slab_dft_dir.is_dir():
                slab_dft_dir.unlink()
            else:
                import shutil

                shutil.rmtree(slab_dft_dir)
        slab_dft_dir.parent.mkdir(parents=True, exist_ok=True)
        slab_dft_dir.symlink_to(slab_external_dir, target_is_directory=True)
        # Guard against a dangling link that would look like a freshly generated
        # input set (exactly the four VASP input files).
        if len(list(slab_dft_dir.iterdir())) == 4:
            raise RuntimeError(
                f"slab_dft_dir={slab_external_dir} looks inaccessible (only 4 entries). "
                "Check the path or the mount; the stage must not silently fall back to "
                "generating local inputs."
            )
        # Take the energy from the external OUTCAR when there is one, and only when that
        # run converged -- a shared slab directory is reused by every molecule, so an
        # unconverged one would poison the whole campaign at once.
        if (slab_external_dir / "OUTCAR").exists():
            energy = _accept_energy("slab", vasp.parse_results(slab_external_dir))
            if energy is None:
                return
            _update_status(task_dir, "dft_slab_energy", energy)
        _update_status(task_dir, "dft_slab_dir", str(slab_external_dir))
        _update_status(task_dir, "dft_slab_done", True)
    elif not status.get("dft_slab_done"):
        slab_atoms = read(mlp_dir / "slab.vasp")
        slab_dft_dir = dft_dir / "slab"

        _generate_inputs(slab_atoms, slab_dft_dir, dft_calc_type)
        if skip_run:
            _update_status(task_dir, "dft_inputs_slab", True)
        else:
            result = vasp.run(slab_dft_dir, vasp_cmd=vasp_cmd)
            if not result["success"]:
                logger.error("DFT slab calculation failed: %s", result.get("error"))
                return
            energy = _accept_energy("slab", vasp.parse_results(slab_dft_dir))
            if energy is None:
                return
            _update_status(task_dir, "dft_slab_energy", energy)
            _update_status(task_dir, "dft_slab_done", True)

    # ---- DFT: the best adslab ------------------------------------------------
    status = _load_json(task_dir / "status.json")
    if not status.get("dft_adslab_done"):
        best_file = mlp_dir / "best.vasp"
        if not best_file.exists():
            logger.error("best.vasp not found in %s", mlp_dir)
            return

        adslab_atoms = read(best_file)
        adslab_dft_dir = dft_dir / "adslab"

        _generate_inputs(adslab_atoms, adslab_dft_dir, dft_calc_type)
        if skip_run:
            _update_status(task_dir, "dft_inputs_adslab", True)
        else:
            result = vasp.run(adslab_dft_dir, vasp_cmd=vasp_cmd)
            if not result["success"]:
                logger.error("DFT adslab calculation failed: %s", result.get("error"))
                return
            energy = _accept_energy("adslab", vasp.parse_results(adslab_dft_dir))
            if energy is None:
                return
            _update_status(task_dir, "dft_adslab_energy", energy)
            _update_status(task_dir, "dft_adslab_done", True)

    # ---- The DFT adsorption energy -------------------------------------------
    status = _load_json(task_dir / "status.json")
    e_mol = status.get("dft_molecule_energy")
    e_slab = status.get("dft_slab_energy")
    e_adslab = status.get("dft_adslab_energy")

    have_all_energies = e_mol is not None and e_slab is not None and e_adslab is not None
    if not skip_run and have_all_energies:
        _update_status(task_dir, "dft_adsorption_energy", e_adslab - e_slab - e_mol)

    # ---- Bader analysis ------------------------------------------------------
    if not skip_run and not status.get("bader_done"):
        adslab_dft_dir = dft_dir / "adslab"
        chgcar = adslab_dft_dir / "CHGCAR"

        if chgcar.exists():
            # AECCAR0 + AECCAR2 is the all-electron reference density; the CHGCAR alone
            # carries only the pseudo valence density, which is why the SCF template sets
            # LAECHG. Falling back to the CHGCAR would give a quietly different number, so
            # it is used only when VASP wrote no AECCARs at all.
            aeccar0 = adslab_dft_dir / "AECCAR0"
            aeccar2 = adslab_dft_dir / "AECCAR2"
            reference = None
            if aeccar0.exists() and aeccar2.exists():
                try:
                    reference = combine_aeccar(aeccar0, aeccar2, adslab_dft_dir / "CHGCAR_sum")
                except RuntimeError as exc:
                    logger.error("could not build the AECCAR reference density: %s", exc)
                    _update_status(task_dir, "bader_error", str(exc))
            else:
                logger.warning(
                    "no AECCAR0/AECCAR2 in %s; running Bader on the pseudo valence density, " "which is less accurate",
                    adslab_dft_dir,
                )

            bader_result = run_bader(chgcar, aeccar_path=reference)

            if bader_result["success"]:
                best_file = mlp_dir / "best.vasp"
                adslab_atoms = read(best_file)
                # A POSCAR carries no ASE tags: reading one back gives every atom tag 0, so
                # the side-car written next to best.vasp is what says where the adsorbate is.
                tags = _load_tags(best_file)
                adsorbate_indices = [index for index, tag in enumerate(tags) if tag == TAG_ADSORBATE]

                if not adsorbate_indices:
                    reason = (
                        f"no adsorbate atoms (tag={TAG_ADSORBATE}) recorded for {best_file}; "
                        "the tags side-car is missing or empty, so the charge transfer cannot "
                        "be attributed. Re-run the MLP stage to write best.vasp with its tags."
                    )
                    logger.error("Bader analysis: %s", reason)
                    _update_status(task_dir, "bader_error", reason)
                elif bader_result["charges"] is None:
                    _update_status(task_dir, "bader_error", "Bader returned no charges")
                elif len(tags) != len(adslab_atoms):
                    reason = f"tags side-car has {len(tags)} entries for {len(adslab_atoms)} atoms"
                    logger.error("Bader analysis: %s", reason)
                    _update_status(task_dir, "bader_error", reason)
                else:
                    charge_transfer = calculate_charge_transfer(
                        bader_result["charges"],
                        adslab_atoms,
                        adsorbate_indices,
                        # The valences of the POTCARs this run actually used, not a guess.
                        valence_electrons=load_zvals(adslab_dft_dir),
                    )
                    _update_status(task_dir, "charge_transfer", charge_transfer["charge_transfer"])
                    _update_status(task_dir, "bader_result", charge_transfer)
                    _update_status(task_dir, "bader_error", None)
                    _update_status(task_dir, "bader_done", True)
            else:
                logger.warning("Bader analysis failed: %s", bader_result.get("error"))
                _update_status(task_dir, "bader_error", bader_result.get("error"))

    if not skip_run:
        if have_all_energies:
            _update_status(task_dir, "dft_done", True)
        else:
            missing = [
                name for name, value in (("molecule", e_mol), ("slab", e_slab), ("adslab", e_adslab)) if value is None
            ]
            reason = f"no usable DFT energy for: {', '.join(missing)}"
            logger.error("DFT stage incomplete: %s", reason)
            _update_status(task_dir, "dft_error", reason)
