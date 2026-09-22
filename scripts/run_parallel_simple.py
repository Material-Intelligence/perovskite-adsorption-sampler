#!/usr/bin/env python
"""Sample and relax a directory of molecules, one shard per SLURM rank.

The script depends on neither the FAIRChem CLI nor Hydra: it reads
``SLURM_PROCID`` and ``SLURM_NTASKS`` directly and slices the molecule list, so
the same command works on one process and on many.

Examples:
    Interactively, with a configuration file::

        salloc -N 1 -G 4 -t 04:00:00 -C gpu -A <YOUR_ACCOUNT> -q interactive
        srun -n 4 --gpus-per-task=1 python scripts/run_parallel_simple.py \\
            --config examples/configs/parallel.yaml

    Or with everything on the command line::

        srun -n 4 --gpus-per-task=1 python scripts/run_parallel_simple.py \\
            --slab data/slabs/FAPbI3_3x3_zcut017.vasp \\
            --molecules-dir data/molecules \\
            --output outputs/parallel_test
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any

import yaml

# Allow running the script straight from a checkout, without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perovml.parallel import SimpleParallelRunner  # noqa: E402

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

#: Options that carry an argparse default, so a value from the configuration file
#: must win over the command line unless the user really typed something else.
DEFAULTED_OPTIONS = frozenset(
    {"pattern", "output", "generator", "pb_num", "pb_cone", "fmax", "steps", "calculator", "save_top"}
)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Lightweight parallel PerovML sampling run")
    parser.add_argument("--config", help="YAML configuration file")
    parser.add_argument("--slab", help="Slab structure file")
    parser.add_argument("--molecules-dir", help="Directory of molecule files")
    parser.add_argument("--pattern", default="*.vasp", help="Glob matching the molecule files")
    parser.add_argument("--output", default="outputs/parallel_run", help="Output directory")
    parser.add_argument(
        "--generator",
        default="Pb_heuristic_sample",
        choices=["Pb_uniform_sample", "Pb_heuristic_sample", "single", "multi"],
        help="Structure generator",
    )
    parser.add_argument("--pb-num", type=int, default=24, help="Number of orientations per molecule")
    parser.add_argument("--pb-cone", type=float, default=30.0, help="Cone half-angle in degrees")
    parser.add_argument("--fmax", type=float, default=0.05, help="Force convergence threshold, eV/A")
    parser.add_argument("--steps", type=int, default=100, help="Maximum optimizer steps")
    parser.add_argument("--calculator", default="mock", choices=["mock", "dpa3"], help="Calculator backend")
    parser.add_argument("--model-path", help="DPA-3 model file path")
    parser.add_argument("--save-top", default="5", help="How many structures to keep, a number or 'all'")
    parser.add_argument("--reduce-only", action="store_true", help="Only merge existing results")
    parser.add_argument(
        "--resume",
        help="Resume an unfinished run directory, e.g. outputs/parallel/<timestamp>",
    )
    return parser.parse_args()


def load_config(args: argparse.Namespace) -> dict[str, Any]:
    """Merge the YAML configuration with the command-line arguments.

    A command-line value overrides the configuration file, except when the option
    was left at its argparse default and the configuration already sets it.

    Args:
        args: Parsed command-line arguments.

    Returns:
        The merged configuration.
    """
    cfg: dict[str, Any] = {}
    if args.config and Path(args.config).exists():
        with open(args.config) as file:
            cfg = yaml.safe_load(file) or {}

    for key in (
        "slab",
        "molecules_dir",
        "pattern",
        "output",
        "generator",
        "pb_num",
        "pb_cone",
        "fmax",
        "steps",
        "calculator",
        "model_path",
        "save_top",
    ):
        cli_val = getattr(args, key.replace("-", "_"), None)

        if key in cfg and key in DEFAULTED_OPTIONS:
            continue

        if cli_val is not None or key not in cfg:
            cfg[key] = cli_val

    return cfg


def create_calculator(cfg: dict[str, Any]) -> tuple[Any, dict[str, Any]]:
    """Build the ASE calculator named by the configuration, once per process.

    This is the same strict factory the CLI uses: an unrecognised ``calculator`` name
    is an error, never a silent fall back to the mock calculator.

    Args:
        cfg: Merged configuration.

    Returns:
        A ``(calculator, provenance)`` tuple; the provenance record goes into
        ``metadata.json``.

    Raises:
        ValueError: If the calculator name is unknown, or a required model path or name
            was not supplied.
    """
    from perovml.calculators.factory import build_calculator

    return build_calculator(cfg)


def create_process_fn(cfg: dict[str, Any], calc):
    """Build the per-molecule callable, closing over the preloaded calculator.

    Args:
        cfg: Merged configuration.
        calc: ASE calculator, loaded once per process.

    Returns:
        A callable taking one task dict and returning its result record.
    """

    def process_molecule(task: dict[str, Any]) -> dict[str, Any]:
        """Sample, relax and rank the placements of one molecule.

        Args:
            task: A task dict with ``sid``, ``slab_file``, ``molecule_file`` and
                ``output_dir``.

        Returns:
            A summary record for this molecule.
        """
        import csv
        import json

        from ase.io import read, write
        from ase.optimize import LBFGS

        from perovml.core.recipes import perov_ml_pipeline, relax_job
        from perovml.utils.structure import build_adsorbate, build_oc_slab_from_atoms

        mol_path = task["molecule_file"]
        slab_path = task["slab_file"]
        sid = task["sid"]
        output_dir = Path(task["output_dir"])

        slab_atoms = read(slab_path)
        slab = build_oc_slab_from_atoms(slab_atoms, min_ab=8.0)

        adsorbate = build_adsorbate(molecule_file=mol_path, run_dir=output_dir)

        def ml_relax_job(atoms):
            """Relax one structure with the preloaded calculator."""
            return relax_job(
                atoms,
                calc=calc,
                optimizer_cls=LBFGS,
                fmax=cfg.get("fmax", 0.05),
                steps=cfg.get("steps", 100),
            )

        # Slab reference: use the precomputed value, or relax it.
        if cfg.get("relaxed_slab_energy") is not None:
            slab_energy = float(cfg["relaxed_slab_energy"])
            slab_relaxed_atoms = read(cfg["relaxed_slab_file"]) if cfg.get("relaxed_slab_file") else slab.atoms
        else:
            slab_relax = ml_relax_job(slab.atoms)
            slab_energy = float(slab_relax["results"]["energy"])
            slab_relaxed_atoms = slab_relax["atoms"]

        # Gas-phase reference: always relaxed, molecules are small and quick.
        gas_relax = ml_relax_job(adsorbate.atoms)
        gas_energy = float(gas_relax["results"]["energy"])

        outputs = perov_ml_pipeline(
            slab=slab,
            adsorbates_kwargs=[{"adsorbate": adsorbate}],
            generator=cfg.get("generator", "Pb_heuristic_sample"),
            ml_relax_job=ml_relax_job,
            pb_num_orientations=cfg.get("pb_num", 24),
            pb_cone_deg=cfg.get("pb_cone", 30.0),
            relaxed_slab_atoms=slab_relaxed_atoms,
            relaxed_slab_energy=slab_energy,
            relaxed_gas_adsorbate_atoms=gas_relax["atoms"],
            relaxed_gas_adsorbate_energy=gas_energy,
        )

        adslabs = outputs["adslabs"]
        top = adslabs[0] if adslabs else None
        energy = float(top["results"]["energy"]) if top else float("inf")
        e_ads = top["results"].get("adsorption_energy") if top else None

        # How many structures to keep.
        save_top = cfg.get("save_top", 5)
        if isinstance(save_top, str) and save_top.lower() == "all":
            n_save = len(adslabs)
        else:
            n_save = int(save_top)
        n_save = min(n_save, len(adslabs))

        mol_dir = output_dir / sid
        mol_dir.mkdir(parents=True, exist_ok=True)

        candidates_data = []
        for rank, result in enumerate(adslabs[:n_save], start=1):
            candidate_e_ads = result["results"].get("adsorption_energy")
            fmax_val = result["results"].get("fmax")
            # The ASE optimizer reports convergence as numpy.bool_, which json cannot encode.
            converged = result["results"].get("converged")

            write(
                str(mol_dir / f"rank_{rank}.vasp"),
                result["atoms"],
                format="vasp",
                direct=True,
                sort=True,
                vasp5=True,
            )

            candidates_data.append(
                {
                    "rank": rank,
                    "energy": float(result["results"]["energy"]),
                    "adsorption_energy": float(candidate_e_ads) if candidate_e_ads is not None else None,
                    "fmax": float(fmax_val) if fmax_val is not None else None,
                    "converged": bool(converged) if converged is not None else None,
                    "anomalies": list(result["results"].get("anomalies", [])),
                    "structure_file": f"rank_{rank}.vasp",
                }
            )

        summary = {
            "molecule": sid,
            "molecule_file": str(mol_path),
            "num_candidates": len(adslabs),
            "num_saved": n_save,
            "slab_energy": slab_energy,
            "gas_energy": gas_energy,
            "candidates": candidates_data,
        }
        with open(mol_dir / "summary.json", "w") as file:
            json.dump(summary, file, indent=2)

        # A flat CSV alongside it, for quick inspection.
        with open(mol_dir / "energies.csv", "w", newline="") as file:
            fieldnames = ["rank", "energy", "adsorption_energy", "fmax", "converged"]
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            for candidate in candidates_data:
                writer.writerow({key: candidate.get(key) for key in fieldnames})

        return {
            "sid": sid,
            "molecule_file": str(mol_path),
            "energy": energy,
            "adsorption_energy": float(e_ads) if e_ads is not None else None,
            "num_candidates": len(adslabs),
            "num_saved": n_save,
            "output_dir": str(mol_dir),
        }

    return process_molecule


def get_completed_molecules(output_dir: Path) -> set[str]:
    """List the molecules that already have a ``summary.json``.

    Args:
        output_dir: Run directory holding one sub-directory per molecule.

    Returns:
        The finished molecule ids.
    """
    return {child.name for child in output_dir.iterdir() if child.is_dir() and (child / "summary.json").exists()}


def reduce_from_summaries(output_dir: Path) -> pd.DataFrame:
    """Aggregate the per-molecule ``summary.json`` files into one table.

    Reading the summaries is more robust than reading the per-rank result files,
    because a molecule's directory is enough on its own.

    Args:
        output_dir: Run directory holding one sub-directory per molecule.

    Returns:
        One row per molecule, describing its best candidate, sorted by adsorption
        energy when that column exists.
    """
    import json

    import pandas as pd

    records = []

    for mol_dir in sorted(output_dir.iterdir()):
        if not mol_dir.is_dir():
            continue

        summary_file = mol_dir / "summary.json"
        if not summary_file.exists():
            continue

        try:
            with open(summary_file) as file:
                summary = json.load(file)

            # Rank 1 is the best candidate.
            candidates = summary.get("candidates", [])
            if candidates:
                best = candidates[0]
                records.append(
                    {
                        "sid": mol_dir.name,
                        "molecule_file": summary.get("molecule_file", ""),
                        "energy": best.get("energy"),
                        "adsorption_energy": best.get("adsorption_energy"),
                        "fmax": best.get("fmax"),
                        "converged": best.get("converged"),
                        "num_candidates": summary.get("num_candidates", len(candidates)),
                        "num_saved": summary.get("num_saved", len(candidates)),
                    }
                )
        except (OSError, ValueError) as exc:
            print(f"[Warn] could not read {summary_file}: {exc}")

    dataframe = pd.DataFrame(records)

    if "adsorption_energy" in dataframe.columns and len(dataframe) > 0:
        dataframe = dataframe.sort_values("adsorption_energy")

    return dataframe


def main() -> None:
    """Shard the molecule list over the available ranks and sample each molecule."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    args = parse_args()
    cfg = load_config(args)

    if args.resume:
        output_dir = Path(args.resume)
        if not output_dir.exists():
            print(f"[Error] resume directory does not exist: {output_dir}")
            return
        # Inherit the previous configuration, so a resumed run stays consistent.
        metadata_file = output_dir / "metadata.json"
        if metadata_file.exists():
            import json

            with open(metadata_file) as file:
                metadata = json.load(file)
            prev_cfg = metadata.get("config", {})
            # The stored value wins for any option the user left at its default.
            for key, value in prev_cfg.items():
                if key in DEFAULTED_OPTIONS or key not in cfg or cfg[key] is None:
                    cfg[key] = value
            print(f"[Info] resuming run: {output_dir}")
            print(f"[Info] calculator={cfg.get('calculator')}, model_path={cfg.get('model_path')}")
    else:
        output_dir = Path(cfg.get("output", "outputs/parallel_run")) / datetime.now().strftime("%Y%m%d_%H%M%S")

    if args.reduce_only:
        print("[Info] merge-only mode")
        base_dir = Path(cfg.get("output", "outputs/parallel_run"))
        # Pick the most recent timestamped run directory.
        runs = sorted(
            (path for path in base_dir.glob("*") if path.is_dir() and path.name[0].isdigit()),
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if runs:
            target_dir = runs[0]
            print(f"[Info] aggregating {target_dir}")
            dataframe = reduce_from_summaries(target_dir)
            if len(dataframe) > 0:
                dataframe.to_csv(target_dir / "combined.csv", index=False)
                dataframe.to_json(target_dir / "combined.json", orient="records", indent=2)
                print(f"[Info] wrote {target_dir}/combined.csv and combined.json")
                print(f"\n{len(dataframe)} molecules in total")
                print(dataframe[["sid", "energy", "adsorption_energy"]].head(10))
        return

    runner = SimpleParallelRunner(output_dir=str(output_dir), result_prefix="perovml")

    molecules_dir = cfg.get("molecules_dir")
    if not molecules_dir:
        print("[Error] --molecules-dir is required")
        return

    mol_files = sorted(glob(str(Path(molecules_dir) / cfg.get("pattern", "*.vasp"))))
    if not mol_files:
        print(f"[Error] no molecule file matched {molecules_dir}/{cfg.get('pattern')}")
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

    if args.resume:
        completed = get_completed_molecules(output_dir)
        tasks = [task for task in all_tasks if task["sid"] not in completed]
        if runner.is_main:
            print(f"[Info] {len(all_tasks)} tasks, {len(completed)} done, {len(tasks)} to run")
        if not tasks:
            print("[Info] every task is already finished")
            return
    else:
        tasks = all_tasks

    # One calculator per process, loaded before the task loop starts -- and before the
    # metadata is written, so an unusable name stops the run rather than filling a
    # directory with results nobody can interpret.
    calc, calculator_provenance = create_calculator(cfg)

    # Only the main process writes the metadata, and only for a new run.
    if runner.is_main and not args.resume:
        runner.save_metadata(
            {
                "config": cfg,
                "calculator_provenance": calculator_provenance,
                "total_tasks": len(tasks),
                "slab_file": slab_file,
                "molecules_dir": molecules_dir,
            }
        )

    process_fn = create_process_fn(cfg, calc)

    runner.run(tasks=tasks, process_fn=process_fn, desc="PerovML")

    if runner.is_main and runner.num_jobs > 1:
        print("\n[Info] once every rank has finished, merge the results with:")
        print(f"  python scripts/run_parallel_simple.py --reduce-only --output {output_dir.parent}")


if __name__ == "__main__":
    main()
