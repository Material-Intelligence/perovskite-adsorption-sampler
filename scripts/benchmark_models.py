#!/usr/bin/env python3
"""Compare two or more DPA-3 model files on a single structure.

For every model the script measures how long it takes to load, how long a single
point takes and how long a relaxation takes, and it compares the energies and the
forces the models produce. The report is written as JSON.

The model files are not distributed with this repository; see ``models/README.md``
for how to obtain them.

Examples:
    python scripts/benchmark_models.py \\
        --structure data/slabs/FAPbI3_3x3_zcut017.vasp \\
        --model DPA-3.1-3M=/path/to/DPA-3.1-3M.pt \\
        --model DPA-3.1-3M-frozen=/path/to/DPA-3.1-3M-Omat24.pth
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Compare two or more DPA-3 model files on one structure.")
    parser.add_argument(
        "--structure",
        required=True,
        help="Structure file to benchmark on (any ASE-readable format).",
    )
    parser.add_argument(
        "--model",
        action="append",
        required=True,
        metavar="NAME=PATH",
        dest="models",
        help="Model to benchmark, as NAME=PATH. Repeat for each model.",
    )
    parser.add_argument("--head", default="Omat24", help="Multi-task head to select (default: Omat24).")
    parser.add_argument("--fmax", type=float, default=0.01, help="Force convergence threshold, eV/A (default: 0.01).")
    parser.add_argument("--steps", type=int, default=100, help="Maximum optimizer steps (default: 100).")
    parser.add_argument(
        "--outdir",
        default="outputs/benchmark_models",
        help="Directory for the JSON report (default: outputs/benchmark_models).",
    )
    return parser.parse_args()


def parse_model_specs(specs: list[str], head: str) -> dict[str, dict[str, str]]:
    """Turn a list of ``NAME=PATH`` strings into a model table.

    Args:
        specs: The strings passed with ``--model``.
        head: Default multi-task head name.

    Returns:
        A mapping of model name to ``{"path": ..., "head": ...}``.

    Raises:
        SystemExit: If an argument is not of the form ``NAME=PATH``.
    """
    models: dict[str, dict[str, str]] = {}
    for spec in specs:
        name, separator, path = spec.partition("=")
        if not separator or not name or not path:
            raise SystemExit(f"--model expects NAME=PATH, got {spec!r}")
        models[name] = {"path": path, "head": head}
    return models


def print_header(title: str) -> None:
    """Print a section title inside a full-width rule.

    Args:
        title: Title text.
    """
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70)


def print_section(title: str) -> None:
    """Print a subsection title.

    Args:
        title: Title text.
    """
    print(f"\n--- {title} ---")


def load_model(model_path: str, head: str = "Omat24") -> tuple[Any, Any]:
    """Load one DPA-3 model and time how long that takes.

    Args:
        model_path: Path to the model file.
        head: Multi-task head to select.

    Returns:
        A tuple of the calculator and the load time in seconds, or ``(None, reason)``
        when the model could not be loaded.
    """
    try:
        from deepmd.calculator import DP as DPCalculator
    except ImportError:
        return None, "deepmd-kit is not installed"

    if not os.path.exists(model_path):
        return None, f"model file not found: {model_path}"

    print(f"[loading] {model_path}")
    print(f"          head = {head}")

    start = time.time()
    try:
        calc = DPCalculator(model_path, head=head)
        return calc, time.time() - start
    except Exception as exc:  # noqa: BLE001 - a failed model is a result, not a crash
        return None, str(exc)


def run_single_point(atoms, calc, n_warmup: int = 1, n_runs: int = 3) -> dict[str, Any]:
    """Time a single-point evaluation, after warming the model up.

    Args:
        atoms: Structure to evaluate. It is copied, never modified.
        calc: ASE calculator.
        n_warmup: Untimed evaluations run first.
        n_runs: Timed evaluations.

    Returns:
        A dict with the energy, the forces, the largest force component and the
        mean, the standard deviation and the individual timings.
    """
    atoms = atoms.copy()
    atoms.calc = calc

    for _ in range(n_warmup):
        atoms.get_potential_energy()
        atoms.get_forces()

    times = []
    energy = None
    forces = None
    for _ in range(n_runs):
        atoms.calc.results = {}  # drop the cache so the work is redone

        start = time.time()
        energy = atoms.get_potential_energy()
        forces = atoms.get_forces()
        times.append(time.time() - start)

    return {
        "energy": energy,
        "forces": forces,
        "fmax": np.abs(forces).max(),
        "time_avg": np.mean(times),
        "time_std": np.std(times),
        "times": times,
    }


def run_optimization(atoms, calc, fmax: float = 0.01, max_steps: int = 100) -> dict[str, Any]:
    """Relax a structure and record the trajectory of energies and forces.

    Args:
        atoms: Structure to relax. It is copied, never modified.
        calc: ASE calculator.
        fmax: Force convergence threshold, in eV/A.
        max_steps: Maximum optimizer steps.

    Returns:
        A dict with the convergence flag, the step count, the timings, the initial
        and final energies and forces, the per-step history and the relaxed structure.
    """
    from ase.optimize import LBFGS

    atoms = atoms.copy()
    atoms.calc = calc

    e_init = atoms.get_potential_energy()
    fmax_init = np.abs(atoms.get_forces()).max()

    start = time.time()
    dyn = LBFGS(atoms, memory=100, damping=1.0, alpha=70.0)

    step_energies: list[float] = []
    step_fmax: list[float] = []

    def record_step() -> None:
        """Append the current energy and largest force to the history."""
        step_energies.append(atoms.get_potential_energy())
        step_fmax.append(np.abs(atoms.get_forces()).max())

    dyn.attach(record_step)

    converged = dyn.run(fmax=fmax, steps=max_steps)
    opt_time = time.time() - start

    e_final = atoms.get_potential_energy()
    fmax_final = np.abs(atoms.get_forces()).max()

    return {
        "converged": converged,
        "n_steps": len(step_energies),
        "time": opt_time,
        "time_per_step": opt_time / max(len(step_energies), 1),
        "e_init": e_init,
        "e_final": e_final,
        "e_change": e_final - e_init,
        "fmax_init": fmax_init,
        "fmax_final": fmax_final,
        "step_energies": step_energies,
        "step_fmax": step_fmax,
        "final_atoms": atoms,
    }


def compare_forces(f1: np.ndarray, f2: np.ndarray, name1: str, name2: str) -> dict[str, float]:
    """Report how far two force sets are apart, and print the comparison.

    Args:
        f1: Forces from the first model.
        f2: Forces from the second model.
        name1: Name of the first model.
        name2: Name of the second model.

    Returns:
        A dict with the mean absolute error, the largest difference and the RMSE,
        all in eV/A.
    """
    diff = f1 - f2
    mae = np.abs(diff).mean()
    max_diff = np.abs(diff).max()
    rmse = np.sqrt((diff**2).mean())

    print(f"\n[force difference] {name1} vs {name2}:")
    print(f"  MAE:      {mae:.6f} eV/A")
    print(f"  max diff: {max_diff:.6f} eV/A")
    print(f"  RMSE:     {rmse:.6f} eV/A")

    return {"mae": mae, "max_diff": max_diff, "rmse": rmse}


def to_serializable(obj: Any) -> Any:
    """Convert NumPy scalars and arrays into JSON-friendly Python objects.

    Args:
        obj: Object to convert; dicts and lists are converted recursively.

    Returns:
        The converted object.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {key: to_serializable(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [to_serializable(value) for value in obj]
    return obj


def main() -> None:
    """Benchmark every model on the given structure and write the JSON report."""
    from ase.io import read

    args = parse_args()
    models = parse_model_specs(args.models, args.head)
    fmax = args.fmax
    max_steps = args.steps

    print_header("DPA-3 model benchmark")
    print(f"time:      {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"structure: {args.structure}")
    print(f"settings:  fmax={fmax}, steps={max_steps}")

    struct_path = Path(args.structure)
    if not struct_path.exists():
        print(f"\n[Error] structure file not found: {struct_path}")
        sys.exit(1)

    atoms = read(str(struct_path))
    print("\n[structure]")
    print(f"  formula: {atoms.get_chemical_formula()}")
    print(f"  atoms:   {len(atoms)}")

    results: dict[str, Any] = {}
    calculators: dict[str, Any] = {}

    # ------------------------------------------------------------------------
    # Step 1: load the models
    # ------------------------------------------------------------------------
    print_header("Step 1: model loading")

    for name, config in models.items():
        print_section(name)

        calc, outcome = load_model(config["path"], config["head"])

        if calc is None:
            print(f"  [failed] {outcome}")
            results[name] = {"load_error": outcome}
        else:
            print(f"  [ok] loaded in {outcome:.3f} s")
            results[name] = {"load_time": outcome}
            calculators[name] = calc

    if not calculators:
        print("\n[Error] no model loaded, nothing to benchmark")
        sys.exit(1)

    # ------------------------------------------------------------------------
    # Step 2: single-point evaluations
    # ------------------------------------------------------------------------
    print_header("Step 2: single-point evaluation")

    single_point_results: dict[str, Any] = {}

    for name, calc in calculators.items():
        print_section(name)

        try:
            sp_result = run_single_point(atoms, calc, n_warmup=1, n_runs=3)
            single_point_results[name] = sp_result
            results[name]["single_point"] = {
                "energy": sp_result["energy"],
                "fmax": sp_result["fmax"],
                "time_avg": sp_result["time_avg"],
                "time_std": sp_result["time_std"],
            }

            print(f"  energy: {sp_result['energy']:.6f} eV")
            print(f"  fmax:   {sp_result['fmax']:.6f} eV/A")
            print(f"  time:   {sp_result['time_avg']:.3f} +/- {sp_result['time_std']:.3f} s")

        except Exception as exc:  # noqa: BLE001 - one failed model must not stop the rest
            print(f"  [failed] {exc}")
            results[name]["single_point_error"] = str(exc)

    if len(single_point_results) == 2:
        names = list(single_point_results.keys())

        print_section("single-point comparison")
        e1 = single_point_results[names[0]]["energy"]
        e2 = single_point_results[names[1]]["energy"]
        print(f"  energy difference ({names[0]} - {names[1]}): {e1 - e2:.6f} eV")

        results["force_diff_single_point"] = compare_forces(
            single_point_results[names[0]]["forces"],
            single_point_results[names[1]]["forces"],
            names[0],
            names[1],
        )

    # ------------------------------------------------------------------------
    # Step 3: relaxations
    # ------------------------------------------------------------------------
    print_header("Step 3: relaxation")
    print(f"settings: fmax={fmax} eV/A, max steps={max_steps}")

    opt_results: dict[str, Any] = {}

    for name, calc in calculators.items():
        print_section(name)

        try:
            opt_result = run_optimization(atoms, calc, fmax=fmax, max_steps=max_steps)
            opt_results[name] = opt_result

            results[name]["optimization"] = {
                "converged": opt_result["converged"],
                "n_steps": opt_result["n_steps"],
                "time": opt_result["time"],
                "time_per_step": opt_result["time_per_step"],
                "e_init": opt_result["e_init"],
                "e_final": opt_result["e_final"],
                "e_change": opt_result["e_change"],
                "fmax_init": opt_result["fmax_init"],
                "fmax_final": opt_result["fmax_final"],
            }

            print(f"  converged:     {'yes' if opt_result['converged'] else 'no'}")
            print(f"  steps:         {opt_result['n_steps']}")
            print(f"  total time:    {opt_result['time']:.2f} s")
            print(f"  time per step: {opt_result['time_per_step']:.3f} s")
            print(f"  energy:        {opt_result['e_init']:.6f} -> {opt_result['e_final']:.6f} eV")
            print(f"                 delta = {opt_result['e_change']:.6f} eV")
            print(f"  fmax:          {opt_result['fmax_init']:.6f} -> {opt_result['fmax_final']:.6f} eV/A")

        except Exception as exc:  # noqa: BLE001 - one failed model must not stop the rest
            import traceback

            print(f"  [failed] {exc}")
            traceback.print_exc()
            results[name]["optimization_error"] = str(exc)

    if len(opt_results) == 2:
        names = list(opt_results.keys())

        print_section("relaxation comparison")

        e1 = opt_results[names[0]]["e_final"]
        e2 = opt_results[names[1]]["e_final"]
        print(f"\n[final energy difference] {names[0]} - {names[1]}: {e1 - e2:.6f} eV")

        results["force_diff_optimized"] = compare_forces(
            opt_results[names[0]]["final_atoms"].get_forces(),
            opt_results[names[1]]["final_atoms"].get_forces(),
            names[0],
            names[1],
        )

        t1 = opt_results[names[0]]["time_per_step"]
        t2 = opt_results[names[1]]["time_per_step"]
        print("\n[speed]")
        print(f"  {names[0]}: {t1:.3f} s/step")
        print(f"  {names[1]}: {t2:.3f} s/step")
        print(f"  ratio ({names[0]}/{names[1]}): {t1 / t2 if t2 > 0 else float('inf'):.2f}x")

    # ------------------------------------------------------------------------
    # Summary and report
    # ------------------------------------------------------------------------
    print_header("Summary")

    for name in models:
        if name not in results:
            continue
        record = results[name]
        print(f"\n[{name}]")

        if "load_error" in record:
            print(f"  load failed: {record['load_error']}")
            continue

        if "load_time" in record:
            print(f"  load time:     {record['load_time']:.3f} s")

        if "single_point" in record:
            single_point = record["single_point"]
            print(f"  single point:  {single_point['energy']:.6f} eV in {single_point['time_avg']:.3f} s")

        if "optimization" in record:
            optimization = record["optimization"]
            status = "converged" if optimization["converged"] else "not converged"
            print(f"  relaxation:    {status} ({optimization['n_steps']} steps, {optimization['time']:.1f} s)")
            print(f"  final energy:  {optimization['e_final']:.6f} eV")

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    # The relaxed Atoms objects are not serialisable, and only live under
    # opt_results, so the per-model records can be written as they are.
    with open(output_file, "w") as file:
        json.dump(to_serializable(results), file, indent=2)

    print(f"\n[Done] report written to {output_file}")
    print("=" * 70)


if __name__ == "__main__":
    main()
