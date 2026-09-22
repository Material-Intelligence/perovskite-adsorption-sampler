#!/usr/bin/env python
"""Relax a directory of gas-phase molecules with an MLP calculator.

The script runs on a single process or across many. Under SLURM it uses
``SLURM_PROCID`` and ``SLURM_NTASKS`` to split the molecule list, falling back to
``mpi4py`` and then to a single process. Each rank writes its own
``results_rank<N>.json``; rank 0 merges them into ``all_results.json`` and
``summary.csv``.

Examples:
    Single process::

        python scripts/optimize_molecules.py --molecules-dir data/molecules \\
            --output outputs/optimized_molecules --calculator mock

    Across 16 ranks::

        srun -n 16 python scripts/optimize_molecules.py --molecules-dir data/molecules \\
            --output outputs/optimized_molecules --calculator dpa3

    With every option spelled out (the model path may also come from
    ``$DPA3_MODEL_PATH``)::

        python scripts/optimize_molecules.py \\
            --molecules-dir data/molecules \\
            --output outputs/optimized_molecules \\
            --calculator dpa3 \\
            --model-path DPA-3.1-3M.pt \\
            --fmax 0.01 \\
            --steps 200
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from ase.io import read, write
from ase.optimize import LBFGS
from tqdm import tqdm

# Allow running the script straight from a checkout, without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perovml.cli.optimize import resolve_dpa3_model_path  # noqa: E402

#: How long rank 0 waits, in units of 0.1 s, for a file another rank has to write.
_FILE_SYNC_POLLS = 600


def get_calculator(calc_type: str, model_path: str | None = None, head: str = "Omat24"):
    """Build the ASE calculator named by ``calc_type``.

    Args:
        calc_type: One of ``"dpa3"``, ``"uma"`` or ``"mock"``.
        model_path: Model file path. For DPA-3 it falls back to
            ``$DPA3_OMAT24_MODEL`` and then ``$DPA3_MODEL_PATH``.
        head: Multi-task head to select, for DPA-3.

    Returns:
        An ASE calculator.

    Raises:
        ValueError: If the calculator type is unknown, or DPA-3 has no model path.
    """
    if calc_type == "dpa3":
        from deepmd.calculator import DP

        return DP(resolve_dpa3_model_path(model_path), head=head)
    if calc_type == "uma":
        from fairchem.core import FAIRChemCalculator

        return FAIRChemCalculator.from_model_checkpoint(
            name_or_path=model_path or os.environ.get("UMA_MODEL") or "uma-m-1p1",
            task_name="omat",
        )
    if calc_type == "mock":
        from perovml.calculators.mock import MockCalculator

        return MockCalculator()
    raise ValueError(f"Unknown calculator: {calc_type}")


def optimize_molecule(
    mol_file: Path,
    output_dir: Path,
    calc,
    fmax: float = 0.01,
    steps: int = 200,
    save_trajectory: bool = False,
) -> dict[str, Any]:
    """Relax one molecule and write the relaxed structure.

    A molecule that fails is reported in the return value rather than raising, so
    one bad structure cannot abort a batch.

    Args:
        mol_file: Molecule file, in any ASE-readable format.
        output_dir: Directory the relaxed POSCAR is written to.
        calc: ASE calculator.
        fmax: Force convergence threshold, in eV/A.
        steps: Maximum optimizer steps.
        save_trajectory: Also write the optimizer trajectory next to the output.

    Returns:
        A record with ``sid``, ``success`` and, on success, the initial and final
        energies, the final fmax, the convergence flag, the step count and the
        elapsed time. On failure it carries ``error`` instead.
    """
    result: dict[str, Any] = {
        "molecule_file": str(mol_file),
        "sid": mol_file.stem,
        "success": False,
        "error": None,
    }

    try:
        atoms = read(mol_file)
        result["initial_energy"] = None
        result["initial_formula"] = atoms.get_chemical_formula()
        result["n_atoms"] = len(atoms)

        atoms.calc = calc

        try:
            result["initial_energy"] = float(atoms.get_potential_energy())
        except Exception:  # noqa: BLE001 - the initial energy is informational only
            pass

        start = time.time()

        traj_file = output_dir / f"{mol_file.stem}_traj.traj" if save_trajectory else None

        dyn = LBFGS(atoms, trajectory=str(traj_file) if traj_file else None)
        converged = dyn.run(fmax=fmax, steps=steps)

        elapsed = time.time() - start

        forces = atoms.get_forces()

        result["final_energy"] = float(atoms.get_potential_energy())
        result["fmax"] = float(np.abs(forces).max())
        result["converged"] = bool(converged)
        result["steps_taken"] = dyn.nsteps
        result["time_seconds"] = elapsed
        result["success"] = True

        output_file = output_dir / f"{mol_file.stem}.vasp"
        atoms.calc = None  # detach the calculator before writing
        write(output_file, atoms, format="vasp")
        result["output_file"] = str(output_file)

    except Exception as exc:  # noqa: BLE001 - one bad molecule must not kill the batch
        result["error"] = str(exc)
        result["success"] = False

    return result


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Relax a directory of molecules with an MLP calculator")
    parser.add_argument("--config", "-c", help="YAML configuration file")
    parser.add_argument("--molecules-dir", help="Directory of molecule files")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--pattern", help="Glob matching the molecule files (default: *.vasp)")
    parser.add_argument("--calculator", choices=["dpa3", "uma", "mock"], help="Calculator backend")
    parser.add_argument("--model-path", help="Model file path or pretrained model name")
    parser.add_argument("--head", default="Omat24", help="DPA-3 multi-task head (default: Omat24)")
    parser.add_argument("--fmax", type=float, help="Force convergence threshold, eV/A")
    parser.add_argument("--steps", type=int, help="Maximum optimizer steps")
    parser.add_argument("--save-trajectory", action="store_true", help="Write the optimizer trajectories")
    parser.add_argument("--resume", action="store_true", help="Skip molecules that already have output")
    parser.add_argument(
        "--gather-timeout",
        type=int,
        default=600,
        help=(
            "Seconds rank 0 waits for the other ranks to write their result files "
            "(SLURM without MPI only; default: 600)"
        ),
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the YAML configuration, the command-line arguments and the defaults.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The merged configuration.
    """
    cfg: dict[str, Any] = {}

    if args.config and Path(args.config).exists():
        with open(args.config) as file:
            cfg = yaml.safe_load(file) or {}

    if args.molecules_dir:
        cfg["molecules_dir"] = args.molecules_dir
    if args.output:
        cfg["output"] = args.output
    if args.pattern:
        cfg["pattern"] = args.pattern
    if args.calculator:
        cfg["calculator"] = args.calculator
    if args.model_path:
        cfg["model_path"] = args.model_path
    if getattr(args, "head", None):
        cfg["head"] = args.head
    if args.fmax:
        cfg["fmax"] = args.fmax
    if args.steps:
        cfg["steps"] = args.steps
    if args.save_trajectory:
        cfg["save_trajectory"] = args.save_trajectory

    cfg.setdefault("pattern", "*.vasp")
    cfg.setdefault("calculator", "dpa3")
    cfg.setdefault("head", "Omat24")
    cfg.setdefault("fmax", 0.01)
    cfg.setdefault("steps", 200)
    cfg.setdefault("save_trajectory", False)
    cfg.setdefault("output", "outputs/optimized_molecules")

    return cfg


def _resolve_ranks() -> tuple[int, int, bool, str]:
    """Work out this process's rank, using SLURM first and mpi4py second.

    SLURM wins because an ``srun`` launch can leave the MPI world degenerate at
    size 1 while the allocation really does have several tasks; the SLURM
    variables still describe the split correctly in that case.

    Returns:
        A tuple of the rank, the world size, whether this is rank 0, and the
        synchronisation mode, one of ``"slurm"``, ``"mpi"`` or ``"single"``.
    """
    slurm_procid = os.environ.get("SLURM_PROCID")
    slurm_ntasks = os.environ.get("SLURM_NTASKS")
    has_slurm = slurm_procid is not None and slurm_ntasks is not None
    sync_mode = "single"

    try:
        from mpi4py import MPI

        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()
        sync_mode = "mpi"
    except ImportError:
        rank = 0
        size = 1

    if has_slurm and int(slurm_ntasks) > 1:
        rank = int(slurm_procid)
        size = int(slurm_ntasks)
        sync_mode = "slurm"

    return rank, size, rank == 0, sync_mode


def _resolve_run_directory(out_root: Path, size: int, sync_mode: str, is_main: bool) -> Path:
    """Agree on one timestamped run directory across every rank.

    With MPI a barrier would do, but a bare SLURM launch has none, so rank 0 writes
    the timestamp to a file keyed on the job id and the other ranks read it back.

    Args:
        out_root: Parent directory for the run.
        size: World size.
        sync_mode: Synchronisation mode from :func:`_resolve_ranks`.
        is_main: Whether this is rank 0.

    Returns:
        The run directory, identical on every rank.

    Raises:
        RuntimeError: If the run-id file does not appear in time.
    """
    if size > 1 and sync_mode == "slurm":
        tag = os.environ.get("SLURM_JOB_ID") or datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id_file = out_root / f"run_id_{tag}.txt"
        if is_main:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_root.mkdir(parents=True, exist_ok=True)
            run_id_file.write_text(timestamp + "\n")
        else:
            for _ in range(_FILE_SYNC_POLLS):  # up to about 60 s
                if run_id_file.exists():
                    timestamp = run_id_file.read_text().strip()
                    break
                time.sleep(0.1)
            else:
                raise RuntimeError(f"Timeout waiting for the run id file: {run_id_file}")
        return out_root / timestamp

    return out_root / datetime.now().strftime("%Y%m%d_%H%M%S")


def main() -> None:
    """Relax every molecule in the configured directory and write the summary."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = parse_args()
    cfg = load_config(args)

    if "molecules_dir" not in cfg:
        print("[Error] set --molecules-dir, or molecules_dir in the configuration file")
        return

    rank, size, is_main, sync_mode = _resolve_ranks()

    molecules_dir = Path(cfg["molecules_dir"])
    mol_files = sorted(molecules_dir.glob(cfg["pattern"]))

    if is_main:
        print(f"[Info] found {len(mol_files)} molecule files")
        print(f"[Info] running on {size} processes")

    if not mol_files:
        if is_main:
            print("[Error] no molecule file found")
        return

    out_root = Path(cfg["output"])
    output_dir = _resolve_run_directory(out_root, size, sync_mode, is_main)

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "metadata.json", "w") as file:
            json.dump(
                {
                    "config": cfg,
                    "timestamp": output_dir.name,
                    "total_molecules": len(mol_files),
                    "num_processes": size,
                },
                file,
                indent=2,
            )

    # Wait for rank 0 to create the directory.
    if size > 1 and sync_mode == "mpi":
        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()
    elif size > 1 and sync_mode == "slurm":
        for _ in range(_FILE_SYNC_POLLS):  # up to about 60 s
            if output_dir.exists():
                break
            time.sleep(0.1)

    completed: set[str] = set()
    if args.resume and output_dir.exists():
        completed = {path.stem for path in output_dir.glob("*.vasp")}
        if is_main:
            print(f"[Info] skipping {len(completed)} molecules that are already done")

    tasks = [path for path in mol_files if path.stem not in completed]

    if is_main:
        print(f"[Info] {len(tasks)} molecules to run")

    if not tasks:
        if is_main:
            print("[Info] every task is already finished")
        return

    # Round-robin split, so every rank gets a similar share.
    my_tasks = [path for index, path in enumerate(tasks) if index % size == rank]

    if is_main:
        print(f"[Info] rank {rank} took {len(my_tasks)} tasks")

    calc = get_calculator(cfg["calculator"], cfg.get("model_path"), head=cfg.get("head", args.head))

    results = []
    iterator = tqdm(my_tasks, desc=f"Rank {rank}", disable=not is_main)

    for mol_file in iterator:
        result = optimize_molecule(
            mol_file=mol_file,
            output_dir=output_dir,
            calc=calc,
            fmax=cfg["fmax"],
            steps=cfg["steps"],
            save_trajectory=cfg.get("save_trajectory", False),
        )
        results.append(result)

        if is_main:
            iterator.set_postfix_str(f"{result['sid']}: {'ok' if result['success'] else 'failed'}")

    with open(output_dir / f"results_rank{rank}.json", "w") as file:
        json.dump(results, file, indent=2)

    # Wait for the other ranks before merging.
    if size > 1 and sync_mode == "mpi":
        from mpi4py import MPI

        MPI.COMM_WORLD.Barrier()
    elif size > 1 and sync_mode == "slurm" and is_main:
        expected = [output_dir / f"results_rank{r}.json" for r in range(size)]
        max_wait = max(0, int(args.gather_timeout))
        waited = 0.0
        while waited < max_wait and not all(path.exists() for path in expected):
            time.sleep(0.5)
            waited += 0.5
        if not all(path.exists() for path in expected):
            missing = [r for r in range(size) if not (output_dir / f"results_rank{r}.json").exists()]
            print(
                f"[Warn] gather timed out after {max_wait}s; no results from ranks {missing}. "
                f"Partial output is in {output_dir}.",
                flush=True,
            )
            return

    if not is_main:
        return

    all_results: list[dict[str, Any]] = []
    for r in range(size):
        result_file = output_dir / f"results_rank{r}.json"
        if result_file.exists():
            with open(result_file) as file:
                all_results.extend(json.load(file))

    with open(output_dir / "all_results.json", "w") as file:
        json.dump(all_results, file, indent=2)

    with open(output_dir / "summary.csv", "w", newline="") as file:
        fieldnames = [
            "sid",
            "success",
            "initial_energy",
            "final_energy",
            "fmax",
            "converged",
            "steps_taken",
            "time_seconds",
            "n_atoms",
        ]
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for record in all_results:
            writer.writerow({key: record.get(key) for key in fieldnames})

    n_success = sum(1 for record in all_results if record["success"])
    n_converged = sum(1 for record in all_results if record.get("converged"))
    total_time = sum(record.get("time_seconds", 0) for record in all_results)

    print()
    print("=" * 60)
    print("  Optimization finished")
    print("=" * 60)
    print(f"  molecules: {len(all_results)}")
    print(f"  succeeded: {n_success}")
    print(f"  converged: {n_converged}")
    print(f"  total time: {total_time:.1f} s")
    print(f"  output directory: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
