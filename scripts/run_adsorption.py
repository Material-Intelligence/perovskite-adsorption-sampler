#!/usr/bin/env python
"""Batch driver for the full adsorption workflow: MLP screening, DFT, Bader.

Layering:
  - this script is the runner layer: it discovers the molecules, shards them over
    the available ranks and reports progress;
  - :func:`perovml.recipes.adsorption.run_adsorption_task` is the recipe layer: it
    handles one molecule and owns the resume logic.

Resuming works at two levels. Within a molecule the recipe layer inspects the
task directory and restarts at the first unfinished step; across molecules this
script skips any task directory whose ``status.json`` already reports completion.

Examples:
    Start a new run::

        srun -n 16 --gpus-per-task=1 python scripts/run_adsorption.py \\
            --config examples/configs/dpa3.yaml

    Resume an existing one by pointing --output at its directory::

        srun -n 16 --gpus-per-task=1 python scripts/run_adsorption.py \\
            --config examples/configs/dpa3.yaml \\
            --output outputs/adsorption/<timestamp>
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from glob import glob
from pathlib import Path
from typing import Any

import yaml

# Allow running the script straight from a checkout, without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perovml.parallel import SimpleParallelRunner  # noqa: E402
from perovml.recipes.adsorption import run_adsorption_task  # noqa: E402
from perovml.utils.structure import build_adsorbate, build_oc_slab_from_atoms  # noqa: E402

# =============================================================================
# Helpers
# =============================================================================


def _load_json(path: Path) -> dict:
    """Read a JSON file, returning an empty dict when it is missing or unreadable.

    Args:
        path: File to read.

    Returns:
        The decoded object, or ``{}`` on any read or decode failure.
    """
    if path.exists():
        try:
            with open(path) as file:
                return json.load(file)
        except (OSError, ValueError):
            return {}
    return {}


def is_task_completed(task_dir: Path, require_dft: bool = False) -> bool:
    """Report whether one molecule's task directory is finished.

    Args:
        task_dir: The molecule's task directory.
        require_dft: Require the DFT stage too, not just the MLP screening.

    Returns:
        True when the requested stage is marked done in ``status.json``.
    """
    status = _load_json(task_dir / "status.json")
    if require_dft:
        return status.get("dft_done", False)
    return status.get("mlp_done", False)


def get_completed_tasks(output_dir: Path, require_dft: bool = False) -> set[str]:
    """Collect the ids of the molecules that are already finished.

    Args:
        output_dir: Run directory holding a ``molecules`` sub-directory.
        require_dft: Require the DFT stage too, not just the MLP screening.

    Returns:
        The finished task ids, empty when the run directory has no molecules yet.
    """
    completed: set[str] = set()
    molecules_dir = output_dir / "molecules"
    if not molecules_dir.exists():
        return completed

    for task_dir in molecules_dir.iterdir():
        if task_dir.is_dir() and is_task_completed(task_dir, require_dft=require_dft):
            completed.add(task_dir.name)

    return completed


# =============================================================================
# Configuration
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Full adsorption workflow (MLP screening plus DFT)")
    parser.add_argument("--config", required=True, help="YAML configuration file")
    parser.add_argument("--slab", help="Override the slab file named in the configuration")
    parser.add_argument("--molecules-dir", help="Override the molecule directory named in the configuration")
    parser.add_argument("--output", help="Output directory; naming an existing one resumes that run")
    parser.add_argument("--no-dft", action="store_true", help="Skip the DFT stage")
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    """Load the YAML configuration and layer the command-line overrides on top.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The merged configuration.
    """
    with open(args.config) as file:
        cfg = yaml.safe_load(file) or {}

    if args.slab:
        cfg["slab"] = args.slab
    if args.molecules_dir:
        cfg["molecules_dir"] = args.molecules_dir
    if args.output:
        cfg["output"] = args.output
    if args.no_dft:
        cfg["run_dft"] = False
    # slab_dft_dir and slab_mlp_relaxed have no command-line flags of their own, but
    # honour them when a caller sets them on the namespace, which helps when debugging.
    if getattr(args, "slab_dft_dir", None):
        cfg["slab_dft_dir"] = args.slab_dft_dir
    if getattr(args, "slab_mlp_relaxed", None) is not None:
        cfg["slab_mlp_relaxed"] = args.slab_mlp_relaxed

    return cfg


def create_calculator(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Build the ASE calculator named by the configuration.

    This is the same strict factory the CLI uses: an unrecognised ``calculator`` name
    is an error, never a silent fall back to the mock calculator.

    Args:
        cfg: Merged configuration. The ``calculator`` key selects the backend.

    Returns:
        A ``(calculator, provenance)`` tuple; the provenance record is written into
        ``metadata.json`` so the results can be traced back to the model that made them.

    Raises:
        ValueError: If the calculator name is unknown, or a required model path or name
            was not supplied.
    """
    from perovml.calculators.factory import build_calculator

    return build_calculator(cfg)


# =============================================================================
# Main
# =============================================================================


def create_process_fn(cfg: dict[str, Any], calc, slab, slab_energy: float | None, slab_relaxed):
    """Build the per-molecule callable the parallel runner applies to each task.

    Args:
        cfg: Merged configuration.
        calc: ASE calculator, loaded once per process.
        slab: The slab every molecule is placed on.
        slab_energy: Relaxed slab energy in eV, shared by every molecule.
        slab_relaxed: Relaxed slab structure, shared by every molecule.

    Returns:
        A callable taking one task dict and returning its result record.
    """

    def process_molecule(task: dict[str, Any]) -> dict[str, Any]:
        """Run the adsorption workflow for one molecule.

        Args:
            task: A task dict with ``sid``, ``molecule_file`` and ``output_dir``.

        Returns:
            The recipe-layer result, with the molecule id and path added.
        """
        sid = task["sid"]
        mol_path = task["molecule_file"]
        output_dir = Path(task["output_dir"])

        task_dir = output_dir / "molecules" / sid

        adsorbate = build_adsorbate(molecule_file=mol_path, run_dir=task_dir)

        # The recipe layer owns the per-molecule resume logic.
        result = run_adsorption_task(
            task_dir=task_dir,
            slab=slab,
            adsorbate=adsorbate,
            calc=calc,
            # Calculation parameters
            fmax=cfg.get("fmax"),
            steps=cfg.get("steps"),
            pb_num_uniform=cfg.get("pb_num_uniform"),
            pb_num_heuristic=cfg.get("pb_num_heuristic"),
            pb_cone=cfg.get("pb_cone"),
            save_top=cfg.get("save_top"),
            # Shared references, so the slab is not relaxed once per molecule
            precomputed_slab_atoms=slab_relaxed,
            precomputed_slab_energy=slab_energy,
            # Provenance
            slab_file=cfg.get("slab"),
            molecule_file=mol_path,
            # DFT stage
            run_dft=cfg.get("run_dft", False),
            dft_mode=cfg.get("dft_mode", "scf"),
            vasp_command=cfg.get("vasp_command", "vasp_std"),
            dft_incar_override=cfg.get("dft_incar_override"),
            disable_anomaly_detection=cfg.get("disable_anomaly_detection", False),
            dft_generate_only=cfg.get("dft_generate_only", False),
            potcar_dir=cfg.get("potcar_dir"),
            slab_mlp_relaxed=cfg.get("slab_mlp_relaxed"),
            slab_dft_dir=cfg.get("slab_dft_dir"),
        )

        return {"sid": sid, "molecule_file": mol_path, **result}

    return process_molecule


def main() -> None:
    """Shard the molecule list over the available ranks and run the workflow."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = parse_args()
    cfg = load_config(args)

    # The configured directory is used as given; no timestamped sub-directory is added.
    output_dir = Path(cfg.get("output", "outputs/adsorption"))

    is_resume = output_dir.exists() and (output_dir / "metadata.json").exists()

    runner = SimpleParallelRunner(output_dir=str(output_dir), result_prefix="adsorption")

    if runner.is_main:
        print(f"[Info] {'resuming' if is_resume else 'starting'} run: {output_dir}")

    molecules_dir = cfg.get("molecules_dir")
    if not molecules_dir:
        print("[Error] the configuration must set molecules_dir")
        return

    pattern = cfg.get("pattern", "*.vasp")
    mol_files = sorted(glob(str(Path(molecules_dir) / pattern)))
    if not mol_files:
        print(f"[Error] no molecule file matched {molecules_dir}/{pattern}")
        return

    slab_file = cfg.get("slab")
    all_tasks = [
        {
            "sid": Path(path).stem,
            "slab_file": slab_file,
            "molecule_file": path,
            "output_dir": output_dir,
        }
        for path in mol_files
    ]

    # Drop the molecules that are already finished.
    require_dft = cfg.get("run_dft", False)
    completed = get_completed_tasks(output_dir, require_dft=require_dft)
    tasks = [task for task in all_tasks if task["sid"] not in completed]

    if runner.is_main:
        print(f"[Info] {len(all_tasks)} tasks, {len(completed)} done, {len(tasks)} to run")

    if not tasks:
        if runner.is_main:
            print("[Info] every task is already finished")
        return

    # One calculator per process. Built before the metadata is written, so that an
    # unusable calculator name stops the run instead of producing a directory of results.
    calc, calculator_provenance = create_calculator(cfg)

    # Only the main process writes the metadata, and only for a new run.
    if runner.is_main and not is_resume:
        runner.save_metadata(
            {
                "config": cfg,
                "calculator_provenance": calculator_provenance,
                "total_tasks": len(all_tasks),
                "slab_file": slab_file,
                "molecules_dir": molecules_dir,
            }
        )

    # Relax the slab once and share it, rather than once per molecule.
    from ase.io import read, write
    from ase.optimize import LBFGS

    from perovml.core.recipes import relax_job

    slab_atoms = read(slab_file)
    slab = build_oc_slab_from_atoms(slab_atoms, min_ab=8.0)

    slab_dir = output_dir / "slab"
    slab_energy = None
    slab_relaxed = None

    if cfg.get("relaxed_slab_energy") is not None:
        slab_energy = float(cfg["relaxed_slab_energy"])
        slab_relaxed = read(cfg["relaxed_slab_file"]) if cfg.get("relaxed_slab_file") else slab.atoms
    elif (slab_dir / "slab_opt.json").exists():
        # Reuse the slab a previous run relaxed.
        slab_data = _load_json(slab_dir / "slab_opt.json")
        if slab_data and "energy" in slab_data and (slab_dir / "slab_opt.vasp").exists():
            slab_energy = slab_data["energy"]
            slab_relaxed = read(slab_dir / "slab_opt.vasp")
        # Otherwise slab_energy stays None and the slab is relaxed below.

    if slab_energy is None:
        slab_result = relax_job(
            slab.atoms,
            calc=calc,
            optimizer_cls=LBFGS,
            fmax=cfg.get("fmax", 0.03),
            steps=cfg.get("steps", 150),
        )
        slab_energy = float(slab_result["results"]["energy"])
        slab_relaxed = slab_result["atoms"]

        if runner.is_main:
            slab_dir.mkdir(parents=True, exist_ok=True)
            write(str(slab_dir / "slab_opt.vasp"), slab_relaxed, format="vasp", direct=True, sort=True, vasp5=True)
            with open(slab_dir / "slab_opt.json", "w") as file:
                json.dump(
                    {"energy": slab_energy, "fmax": float(slab_result["results"].get("fmax", 0))},
                    file,
                    indent=2,
                )

    process_fn = create_process_fn(cfg, calc, slab, slab_energy, slab_relaxed)

    results = runner.run(tasks=tasks, process_fn=process_fn, desc="Adsorption")

    if runner.is_main:
        print(f"\n[Info] finished {len(results)} molecules in this pass")
        total_completed = len(get_completed_tasks(output_dir))
        print(f"[Info] overall progress: {total_completed}/{len(all_tasks)}")


if __name__ == "__main__":
    main()
