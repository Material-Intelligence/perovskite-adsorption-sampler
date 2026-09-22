"""PerovML pipeline (applications layer).

This mirrors the structure of ``fairchem.core.components.calculate.recipes.adsorbml``
but routes structure generation through the PerovML samplers (including the Pb
modes) while remaining compatible with the legacy single/multi samplers.
"""

from __future__ import annotations

import warnings
from functools import partial
from typing import Any, Literal

import numpy as np
from ase.atoms import Atoms
from fairchem.core.components.calculate.recipes.adsorbml import detect_anomaly
from fairchem.data.oc.core.adsorbate import Adsorbate
from fairchem.data.oc.core.adsorbate_slab_config import AdsorbateSlabConfig
from fairchem.data.oc.core.multi_adsorbate_slab_config import MultipleAdsorbateSlabConfig
from fairchem.data.oc.core.slab import Slab

from perovml.core.placement import PbAdsorbateSlabConfig

__all__ = [
    "ANOMALY_SEVERITY",
    "dual_sampling_pipeline",
    "get_anomaly_severity",
    "perov_adslab_generator",
    "perov_ml_pipeline",
    "relax_job",
    "run_perovml",
    "select_best_by_priority",
]

GeneratorName = Literal["single", "multi", "Pb_uniform_sample", "Pb_heuristic_sample"]
SamplingMode = Literal["random", "heuristic", "random_site_heuristic_placement"]


def relax_job(initial_atoms: Atoms, calc, optimizer_cls, fmax: float, steps: int) -> dict[str, Any]:
    """Relax one structure and report the energy, forces and convergence state.

    The calculator may be a mock, which is what the tests use.

    Args:
        initial_atoms: Structure to relax. It is copied, never modified.
        calc: ASE calculator used for energies and forces.
        optimizer_cls: ASE optimizer class, e.g. ``LBFGS``.
        fmax: Force convergence threshold, in eV/A.
        steps: Maximum number of optimizer steps.

    Returns:
        A dict with ``input_atoms``, ``atoms`` and ``results``, shaped like the
        return value of ``fairchem``'s own ``relax_job``.
    """
    atoms = initial_atoms.copy()
    # Normalise the periodic boundary flags. A mixed state such as (True, False, ...)
    # makes UMA and ASE raise "Attempted to guess PBC ... some dimensions but not others".
    pbc = atoms.get_pbc()
    if pbc.any() and not pbc.all():
        # Promote anything partially periodic to fully periodic. Gas-phase molecules
        # normally sit in a large cell, so this is geometrically harmless for them too.
        atoms.set_pbc((True, True, True))
    atoms.calc = calc
    dyn = optimizer_cls(atoms)
    converged = dyn.run(fmax=fmax, steps=steps)
    nsteps = getattr(dyn, "nsteps", None)
    try:
        nsteps = int(nsteps) if nsteps is not None else None
    except (TypeError, ValueError):
        nsteps = None
    energy = atoms.get_potential_energy()
    forces = atoms.get_forces()
    fmax_final = float(np.abs(forces).max())
    atoms.calc = None
    return {
        "input_atoms": {"atoms": initial_atoms},
        "atoms": atoms,
        "results": {
            "energy": energy,
            "forces": forces,
            "fmax": fmax_final,
            "converged": converged,
            "nsteps": nsteps,
        },
    }


def perov_adslab_generator(
    slab: Slab | Atoms,
    adsorbates_kwargs: list[dict[str, Any]] | None = None,
    generator: GeneratorName = "single",
    num_sites: int = 100,
    num_augmentations: int = 1,
    mode: SamplingMode = "random",
    num_configurations: int = 1,
    pb_num_orientations: int = 50,
    pb_cone_deg: float = 20.0,
    interstitial_gap: float = 0.1,
    rng: np.random.Generator | int | None = None,
) -> list[Atoms]:
    """Generate adslab candidate structures with a legacy or a PerovML sampler.

    Args:
        slab: Slab structure. Plain ``Atoms`` are wrapped in ``Slab(slab_atoms=...)``.
        adsorbates_kwargs: One spec per adsorbate, following the FAIRChem convention,
            e.g. ``{"adsorbate_smiles_from_db": "*O"}``. Defaults to a single ``*O``.
        generator: Which sampler to use. ``single`` and ``multi`` are the FAIRChem
            samplers; the ``Pb_*`` modes are the PerovML ones.
        num_sites: Number of adsorption sites, for the ``single``/``multi`` samplers.
        num_augmentations: Augmentations per site, for the ``single`` sampler.
        mode: Site/placement mode, for the ``single``/``multi`` samplers.
        num_configurations: Number of configurations, for the ``multi`` sampler.
        pb_num_orientations: Number of orientations, for the ``Pb_*`` samplers.
        pb_cone_deg: Cone half-angle in degrees, for ``Pb_heuristic_sample``.
        interstitial_gap: Extra clearance in Angstrom kept when lifting the molecule
            along the surface normal, for the ``Pb_*`` samplers.
        rng: Seed or generator for ``Pb_uniform_sample``; see
            :class:`~perovml.core.placement.PbAdsorbateSlabConfig`. An integer makes the
            candidate set reproducible.

    Returns:
        The generated adslab structures.

    Raises:
        ValueError: If an adsorbate spec or the generator name is not recognised.
    """
    if isinstance(slab, Atoms):
        slab = Slab(slab_atoms=slab)

    # Build the Adsorbate objects.
    adsorbates: list[Adsorbate] = []
    for kwargs in adsorbates_kwargs or [{"adsorbate_smiles_from_db": "*O"}]:
        if isinstance(kwargs, Adsorbate):
            adsorbates.append(kwargs)
            continue
        if "adsorbate" in kwargs and isinstance(kwargs["adsorbate"], Adsorbate):
            adsorbates.append(kwargs["adsorbate"])
            continue
        if "adsorbate_smiles_from_db" in kwargs:
            adsorbates.append(Adsorbate(adsorbate_smiles_from_db=kwargs["adsorbate_smiles_from_db"]))
            continue
        if "adsorbate_atoms" in kwargs:
            # Preserve the binding indices when they are supplied alongside the atoms.
            if kwargs.get("adsorbate_binding_indices") is not None:
                adsorbates.append(
                    Adsorbate(
                        adsorbate_atoms=kwargs["adsorbate_atoms"],
                        adsorbate_binding_indices=list(kwargs["adsorbate_binding_indices"]),
                    )
                )
            else:
                adsorbates.append(Adsorbate(adsorbate_atoms=kwargs["adsorbate_atoms"]))
            continue
        raise ValueError(
            "Unsupported adsorbate spec; provide an Adsorbate or "
            "{'adsorbate_smiles_from_db'|'adsorbate_atoms'[, 'adsorbate_binding_indices']}"
        )

    if generator == "single":
        return AdsorbateSlabConfig(
            slab,
            adsorbates[0],
            num_sites=num_sites,
            num_augmentations_per_site=num_augmentations,
            mode=mode,
        ).atoms_list
    if generator == "multi":
        return MultipleAdsorbateSlabConfig(
            slab,
            adsorbates,
            num_sites=num_sites,
            num_configurations=num_configurations,
            mode=mode,
        ).atoms_list
    if generator == "Pb_uniform_sample":
        return PbAdsorbateSlabConfig(
            slab=slab,
            adsorbate=adsorbates[0],
            num_orientations=pb_num_orientations,
            interstitial_gap=interstitial_gap,
            mode="Pb_uniform_sample",
            rng=rng,
        ).atoms_list
    if generator == "Pb_heuristic_sample":
        return PbAdsorbateSlabConfig(
            slab=slab,
            adsorbate=adsorbates[0],
            num_orientations=pb_num_orientations,
            cone_angle_deg=pb_cone_deg,
            interstitial_gap=interstitial_gap,
            mode="Pb_heuristic_sample",
        ).atoms_list
    raise ValueError(f"Unknown generator: {generator}")


def perov_ml_pipeline(
    slab: Slab,
    adsorbates_kwargs: list[dict[str, Any]],
    generator: str,
    ml_relax_job,
    pb_num_orientations: int = 50,
    pb_cone_deg: float = 20.0,
    num_sites: int = 100,
    num_augmentations: int = 1,
    num_configurations: int = 1,
    mode: SamplingMode = "random",
    interstitial_gap: float = 0.1,
    rng: np.random.Generator | int | None = None,
    relaxed_slab_atoms: Atoms | None = None,
    relaxed_slab_energy: float | None = None,
    relaxed_gas_adsorbate_atoms: Atoms | None = None,
    relaxed_gas_adsorbate_energy: float | None = None,
) -> dict[str, Any]:
    """Generate, relax and rank adslab candidates.

    Args:
        slab: Slab to place the adsorbate on.
        adsorbates_kwargs: Adsorbate specs, see :func:`perov_adslab_generator`.
        generator: Sampler name, see :func:`perov_adslab_generator`.
        ml_relax_job: Callable taking one ``Atoms`` and returning a ``relax_job``-shaped dict.
        pb_num_orientations: Number of orientations, for the ``Pb_*`` samplers.
        pb_cone_deg: Cone half-angle in degrees, for ``Pb_heuristic_sample``.
        num_sites: Number of adsorption sites, for the ``single``/``multi`` samplers.
        num_augmentations: Augmentations per site, for the ``single`` sampler.
        num_configurations: Number of configurations, for the ``multi`` sampler.
        mode: Site/placement mode, for the ``single``/``multi`` samplers.
        interstitial_gap: Extra clearance in Angstrom kept when lifting the molecule,
            for the ``Pb_*`` samplers.
        rng: Seed or generator for ``Pb_uniform_sample``. An integer makes the
            candidate set reproducible.
        relaxed_slab_atoms: Relaxed bare slab. Accepted for interface symmetry;
            anomaly detection uses the slab part of the initial adslab instead.
        relaxed_slab_energy: Relaxed bare-slab energy, in eV.
        relaxed_gas_adsorbate_atoms: Relaxed gas-phase adsorbate. Unused here; kept
            so that callers can pass the full reference set in one call.
        relaxed_gas_adsorbate_energy: Relaxed gas-phase adsorbate energy, in eV.

    Returns:
        A dict with ``slab`` metadata, the ranked ``adslabs``, the per-candidate
        ``adslab_anomalies`` **in the same order as** ``adslabs`` and the
        ``references`` that were used. Candidates are sorted by adsorption energy when
        both references were given, otherwise by raw energy, ascending.
    """
    # 1) Generate adslab candidates.
    adslab_atoms_list = perov_adslab_generator(
        slab,
        adsorbates_kwargs=adsorbates_kwargs,
        generator=generator,  # 'single' | 'multi' | 'Pb_*'
        num_sites=num_sites,
        num_augmentations=num_augmentations,
        num_configurations=num_configurations,
        pb_num_orientations=pb_num_orientations,
        pb_cone_deg=pb_cone_deg,
        interstitial_gap=interstitial_gap,
        rng=rng,
        mode=mode,
    )

    # 2) Relax all of them.
    results_relaxed = []
    for init_atoms in adslab_atoms_list:
        result = ml_relax_job(init_atoms)
        final_atoms = result["atoms"]
        # final_slab_atoms=None is FAIRChem's own default: compare against the slab part of
        # the initial adslab. A separately relaxed bare slab differs from it by more than the
        # detector's threshold, which marked every candidate as surface_changed.
        anomalies = detect_anomaly(init_atoms, final_atoms, final_slab_atoms=None)

        # Adsorption energy, only when both references are known.
        e_ads = None
        if relaxed_slab_energy is not None and relaxed_gas_adsorbate_energy is not None:
            e_ads = float(result["results"]["energy"]) - relaxed_slab_energy - relaxed_gas_adsorbate_energy
        result["results"]["adsorption_energy"] = e_ads
        result["results"]["anomalies"] = anomalies
        results_relaxed.append(result)

    def sort_key(record: dict[str, Any]) -> float:
        """Sort by adsorption energy when it is known, otherwise by raw energy."""
        e_ads = record["results"].get("adsorption_energy")
        return e_ads if e_ads is not None else record["results"]["energy"]

    results_sorted = sorted(results_relaxed, key=sort_key)

    return {
        "slab": slab.get_metadata_dict(),
        "adslabs": results_sorted,
        # Built from the sorted list, so that adslab_anomalies[i] belongs to adslabs[i].
        # Each record also carries its own copy under results["anomalies"].
        "adslab_anomalies": [record["results"]["anomalies"] for record in results_sorted],
        "references": {
            "slab_energy": relaxed_slab_energy,
            "gas_adsorbate_energy": relaxed_gas_adsorbate_energy,
        },
    }


def run_perovml(
    slab: Slab | Atoms,
    adsorbate: str | Adsorbate | Atoms,
    calculator,
    optimizer_cls,
    fmax: float = 0.02,
    steps: int = 300,
    generator: str = "Pb_uniform_sample",
    num_sites: int = 100,
    num_augmentations: int = 1,
    num_configurations: int = 1,
    pb_num_orientations: int = 50,
    pb_cone_deg: float = 20.0,
    interstitial_gap: float = 0.1,
    rng: np.random.Generator | int | None = None,
    mode: SamplingMode = "random",
    precomputed_slab_atoms: Atoms | None = None,
    precomputed_slab_energy: float | None = None,
    precomputed_gas_atoms: Atoms | None = None,
    precomputed_gas_energy: float | None = None,
) -> dict[str, Any]:
    """Relax the references, then run :func:`perov_ml_pipeline` on them.

    This is the convenience wrapper that mirrors FAIRChem's ``run_adsorbml``.

    Args:
        slab: Slab structure, either ``Atoms`` or a FAIRChem ``Slab``.
        adsorbate: Adsorbate database key, ``Adsorbate`` object, or ``Atoms``.
        calculator: ASE calculator for energy and force evaluation.
        optimizer_cls: ASE optimizer class, e.g. ``LBFGS``, ``BFGS`` or ``FIRE``.
        fmax: Force convergence threshold, in eV/A.
        steps: Maximum number of optimizer steps.
        generator: Structure generator name.
        num_sites: Number of adsorption sites to sample.
        num_augmentations: Augmentations per site.
        num_configurations: Number of configurations, for the multi-adsorbate sampler.
        pb_num_orientations: Number of orientations, for the ``Pb_*`` samplers.
        pb_cone_deg: Cone half-angle in degrees, for ``Pb_heuristic_sample``.
        interstitial_gap: Extra clearance in Angstrom kept when lifting the molecule,
            for the ``Pb_*`` samplers.
        rng: Seed or generator for ``Pb_uniform_sample``. Pass an integer to make the
            run reproducible, and record it alongside the results.
        mode: Sampling mode.
        precomputed_slab_atoms: Pre-relaxed slab. Skips the slab relaxation.
        precomputed_slab_energy: Pre-computed slab energy in eV. Required to skip
            the slab relaxation.
        precomputed_gas_atoms: Pre-relaxed gas-phase adsorbate.
        precomputed_gas_energy: Pre-computed gas-phase energy in eV. Required to
            skip the gas-phase relaxation.

    Returns:
        The :func:`perov_ml_pipeline` output, with the relaxed reference structures
        added under ``references``.

    Raises:
        ValueError: If ``adsorbate`` is none of the three accepted types.
    """
    ml_relax_job = partial(relax_job, calc=calculator, optimizer_cls=optimizer_cls, fmax=fmax, steps=steps)

    adsorbates_kwargs: list[dict[str, Any]] = []
    if isinstance(adsorbate, str):
        adsorbates_kwargs.append({"adsorbate_smiles_from_db": adsorbate})
    elif isinstance(adsorbate, Adsorbate):
        adsorbates_kwargs.append({"adsorbate": adsorbate})
    elif isinstance(adsorbate, Atoms):
        adsorbates_kwargs.append({"adsorbate_atoms": adsorbate})
    else:
        raise ValueError("adsorbate must be a database key, an Adsorbate, or Atoms")

    # Slab reference: use the precomputed values, or relax.
    if precomputed_slab_energy is not None:
        slab_energy = float(precomputed_slab_energy)
        slab_relaxed_atoms = (
            precomputed_slab_atoms
            if precomputed_slab_atoms is not None
            else (slab if isinstance(slab, Atoms) else slab.atoms)
        )
    else:
        slab_relax = ml_relax_job(slab if isinstance(slab, Atoms) else slab.atoms)
        slab_energy = float(slab_relax["results"]["energy"])
        slab_relaxed_atoms = slab_relax["atoms"]

    # Gas-phase adsorbate reference: use the precomputed values, or relax.
    if precomputed_gas_energy is not None:
        gas_energy = float(precomputed_gas_energy)
        gas_relaxed_atoms = precomputed_gas_atoms
        # Without an explicit structure, rebuild one so that it can still be reported.
        if gas_relaxed_atoms is None:
            if isinstance(adsorbate, Atoms):
                gas_relaxed_atoms = adsorbate
            elif isinstance(adsorbate, Adsorbate):
                gas_relaxed_atoms = adsorbate.atoms
            else:
                gas_relaxed_atoms = Adsorbate(adsorbate_smiles_from_db=adsorbate).atoms
    else:
        gas_atoms = None
        if isinstance(adsorbate, Atoms):
            gas_atoms = adsorbate
        elif isinstance(adsorbate, Adsorbate):
            gas_atoms = adsorbate.atoms
        if gas_atoms is None:
            # The adsorbate is a database key, so a gas-phase molecule still has to be built.
            gas_atoms = Adsorbate(adsorbate_smiles_from_db=adsorbate).atoms
        gas_relax = ml_relax_job(gas_atoms)
        gas_energy = float(gas_relax["results"]["energy"])
        gas_relaxed_atoms = gas_relax["atoms"]

    outputs = perov_ml_pipeline(
        slab=slab if isinstance(slab, Slab) else Slab(slab_atoms=slab),
        adsorbates_kwargs=adsorbates_kwargs,
        generator=generator,
        ml_relax_job=ml_relax_job,
        pb_num_orientations=pb_num_orientations,
        pb_cone_deg=pb_cone_deg,
        interstitial_gap=interstitial_gap,
        rng=rng,
        num_sites=num_sites,
        num_augmentations=num_augmentations,
        num_configurations=num_configurations,
        mode=mode,
        relaxed_slab_atoms=slab_relaxed_atoms,
        relaxed_slab_energy=slab_energy,
        relaxed_gas_adsorbate_atoms=gas_relaxed_atoms,
        relaxed_gas_adsorbate_energy=gas_energy,
    )
    outputs["references"].update(
        {
            "slab_atoms": slab_relaxed_atoms,
            "gas_adsorbate_atoms": gas_relaxed_atoms,
        }
    )
    return outputs


# -----------------------------------------------------------------------------
# Anomaly-aware ranking
# -----------------------------------------------------------------------------

ANOMALY_SEVERITY: dict[str, int] = {
    "adsorbate_desorbed": 1,  # desorption, the mildest case
    "adsorbate_dissociated": 2,  # dissociation
    "surface_changed": 2,  # surface reconstruction
    "adsorbate_intercalated": 2,  # intercalation
}
"""How bad each AdsorbML anomaly is; a smaller number is milder."""


def get_anomaly_severity(anomalies: list[str]) -> int:
    """Return the worst severity in an anomaly list.

    Args:
        anomalies: Anomaly names as reported by ``detect_anomaly``.

    Returns:
        0 when the list is empty, otherwise the largest severity found. Unknown
        anomaly names count as severity 2.
    """
    if not anomalies:
        return 0
    return max(ANOMALY_SEVERITY.get(anomaly, 2) for anomaly in anomalies)


def select_best_by_priority(results: list[dict[str, Any]], n: int = 1) -> list[dict[str, Any]]:
    """Rank relaxed candidates by anomaly severity first, then by energy.

    The ordering is: no anomaly, then desorption only, then the more serious
    anomalies. Within one severity level the candidates are sorted by adsorption
    energy when it is known, otherwise by raw energy, ascending.

    Args:
        results: Relaxed candidates, as returned in ``perov_ml_pipeline``'s ``adslabs``.
        n: How many candidates to return.

    Returns:
        The best ``n`` candidates, best first. Empty if ``results`` is empty.
    """
    if not results:
        return []

    def sort_key(record: dict[str, Any]) -> tuple[int, float]:
        anomalies = record["results"].get("anomalies", [])
        severity = get_anomaly_severity(anomalies)
        e_ads = record["results"].get("adsorption_energy")
        energy = e_ads if e_ads is not None else record["results"]["energy"]
        return (severity, energy)

    return sorted(results, key=sort_key)[:n]


# -----------------------------------------------------------------------------
# Dual sampling
# -----------------------------------------------------------------------------


def dual_sampling_pipeline(
    slab: Slab,
    adsorbates_kwargs: list[dict[str, Any]],
    ml_relax_job,
    pb_num_uniform: int = 30,
    pb_num_heuristic: int = 12,
    pb_cone_deg: float = 25.0,
    relaxed_slab_atoms: Atoms | None = None,
    relaxed_slab_energy: float | None = None,
    relaxed_gas_adsorbate_energy: float | None = None,
) -> dict[str, Any]:
    """Run the uniform and the heuristic Pb sampler and merge their candidates.

    Deprecated:
        Use :func:`perovml.recipes.adsorption.generate_dual_sampling_configs` with
        :func:`perovml.recipes.adsorption.relax_single_config` instead, or
        :func:`perovml.recipes.adsorption.run_adsorption_task` for the whole workflow.
        Those are resumable, name every candidate and keep their state on disk. This
        function is kept only so that existing callers do not break.

    Args:
        slab: Slab to place the adsorbate on.
        adsorbates_kwargs: Adsorbate specs, see :func:`perov_adslab_generator`.
        ml_relax_job: Callable taking one ``Atoms`` and returning a ``relax_job``-shaped dict.
        pb_num_uniform: Number of uniformly sampled orientations.
        pb_num_heuristic: Number of cone-sampled orientations.
        pb_cone_deg: Cone half-angle in degrees for the heuristic sampler.
        relaxed_slab_atoms: Relaxed bare slab. Accepted for interface symmetry; anomaly
            detection uses the slab part of the initial adslab instead.
        relaxed_slab_energy: Relaxed bare-slab energy, in eV.
        relaxed_gas_adsorbate_energy: Relaxed gas-phase adsorbate energy, in eV.

    Returns:
        A dict with ``slab`` metadata, the merged and ranked ``adslabs``, the
        per-candidate ``adslab_anomalies`` in the same order as ``adslabs``, the
        ``references`` and a ``sampling_stats`` breakdown per sampler.
    """
    warnings.warn(
        "dual_sampling_pipeline is deprecated; use perovml.recipes.adsorption."
        "generate_dual_sampling_configs with relax_single_config, or run_adsorption_task.",
        DeprecationWarning,
        stacklevel=2,
    )

    all_results: list[dict[str, Any]] = []

    # 1) Uniform sampling.
    uniform_atoms = perov_adslab_generator(
        slab,
        adsorbates_kwargs=adsorbates_kwargs,
        generator="Pb_uniform_sample",
        pb_num_orientations=pb_num_uniform,
    )

    # 2) Heuristic sampling.
    heuristic_atoms = perov_adslab_generator(
        slab,
        adsorbates_kwargs=adsorbates_kwargs,
        generator="Pb_heuristic_sample",
        pb_num_orientations=pb_num_heuristic,
        pb_cone_deg=pb_cone_deg,
    )

    # 3) Merge, keeping track of which sampler produced each candidate.
    all_configs: list[dict[str, Any]] = []
    for index, atoms in enumerate(uniform_atoms):
        all_configs.append({"atoms": atoms, "source": "uniform", "index": index})
    for index, atoms in enumerate(heuristic_atoms):
        all_configs.append({"atoms": atoms, "source": "heuristic", "index": index})

    # 4) Relax every candidate.
    for config in all_configs:
        init_atoms = config["atoms"]
        result = ml_relax_job(init_atoms)
        final_atoms = result["atoms"]

        # See perov_ml_pipeline: the slab part of the initial adslab is the reference.
        anomalies = detect_anomaly(init_atoms, final_atoms, final_slab_atoms=None)

        e_ads = None
        if relaxed_slab_energy is not None and relaxed_gas_adsorbate_energy is not None:
            e_ads = float(result["results"]["energy"]) - relaxed_slab_energy - relaxed_gas_adsorbate_energy

        result["results"]["adsorption_energy"] = e_ads
        result["results"]["anomalies"] = anomalies
        result["results"]["source"] = config["source"]
        result["results"]["source_index"] = config["index"]

        all_results.append(result)

    # 5) Rank by anomaly priority.
    sorted_results = select_best_by_priority(all_results, n=len(all_results))

    return {
        "slab": slab.get_metadata_dict(),
        "adslabs": sorted_results,
        # In the same order as adslabs; each record also carries results["anomalies"].
        "adslab_anomalies": [record["results"]["anomalies"] for record in sorted_results],
        "references": {
            "slab_energy": relaxed_slab_energy,
            "gas_adsorbate_energy": relaxed_gas_adsorbate_energy,
        },
        "sampling_stats": {
            "uniform_count": len(uniform_atoms),
            "heuristic_count": len(heuristic_atoms),
            "total_count": len(all_configs),
        },
    }
