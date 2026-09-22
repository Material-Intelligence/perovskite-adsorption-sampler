r"""Generate adsorbate placements on a slab, without evaluating any energies.

This is the sampling half of the pipeline on its own: it adapts a raw slab into a
FAIRChem ``Slab``, builds the adsorbate, and writes every generated adsorbate-slab
structure as a POSCAR. Useful for inspecting what a sampler produces before
spending compute on relaxations.

Examples:
    perovml place --slab-file data/slabs/FAPbI3_3x3_zcut017.vasp \\
        --molecule-file data/molecules/Acetone.vasp \\
        --mode Pb_heuristic_sample --pb-num-orientations 12 --out-root outputs/placements
"""

from __future__ import annotations

import argparse

PLACEMENT_MODES = (
    "random",
    "heuristic",
    "random_site_heuristic_placement",
    "Pb_uniform_sample",
    "Pb_heuristic_sample",
)

_CONFIG_KEYS = (
    "slab_file",
    "slab_dir",
    "pattern",
    "molecule_file",
    "adsorbate",
    "binding_index",
    "which",
    "mode",
    "num_sites",
    "num_augmentations",
    "num_configurations",
    "copies",
    "pb_num_orientations",
    "pb_cone_deg",
    "interstitial_gap",
    "surface_fraction",
    "min_ab",
    "out_root",
    "seed",
)


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``perovml place`` arguments.

    Args:
        parser: Subcommand parser to populate.
    """
    # Slab source: a single file or a directory of files
    slab_grp = parser.add_mutually_exclusive_group()
    slab_grp.add_argument("--slab-file", help="Slab file (any ASE-readable format)")
    slab_grp.add_argument("--slab-dir", help="Directory containing slab files")
    parser.add_argument("--pattern", default="*.vasp", help="Glob pattern under --slab-dir (default: *.vasp)")

    # Molecule source: a file or an adsorbate database key
    mol_grp = parser.add_mutually_exclusive_group()
    mol_grp.add_argument("--molecule-file", help="Adsorbate molecule file (any ASE-readable format)")
    mol_grp.add_argument("--adsorbate", help="Adsorbate key from the FAIRChem database, e.g. '*O'")
    parser.add_argument("--binding-index", type=int, help="Binding atom index; guessed when omitted")

    parser.add_argument(
        "--which",
        choices=["single", "multiple"],
        default="single",
        help="AdsorbateSlabConfig (single) or MultipleAdsorbateSlabConfig (multiple)",
    )
    parser.add_argument(
        "--mode",
        choices=list(PLACEMENT_MODES),
        default="random_site_heuristic_placement",
        help="Placement mode when --which single (includes the custom Pb_* modes)",
    )
    parser.add_argument("--num-sites", type=int, default=50, help="Number of sites to sample (single)")
    parser.add_argument("--num-augmentations", type=int, default=1, help="Augmentations per site (single)")
    parser.add_argument("--num-configurations", type=int, default=100, help="Number of configurations (multiple)")
    parser.add_argument("--copies", type=int, default=1, help="Adsorbate copies when --which multiple")

    parser.add_argument("--pb-num-orientations", type=int, default=50, help="Orientations for the Pb_* modes")
    parser.add_argument("--pb-cone-deg", type=float, default=20.0, help="Cone half-angle (deg) for Pb_heuristic_sample")
    parser.add_argument(
        "--interstitial-gap",
        type=float,
        default=0.1,
        help="Extra clearance (A) above the covalent contact distance when lifting the molecule (Pb_* modes)",
    )

    parser.add_argument(
        "--surface-fraction",
        type=float,
        default=0.25,
        help="Fallback when the slab has no tags and no Pb-I surface layer is found: "
        "mark this top fraction (by z) as surface",
    )
    parser.add_argument("--min-ab", type=float, default=8.0, help="Minimum a,b cell length required by Slab")

    parser.add_argument("--config", help="YAML config file supplying any of the options above")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for deterministic sampling")
    parser.add_argument(
        "--out-root",
        default="outputs",
        help="Directory the timestamped run folder is created in (default: outputs)",
    )


def _merge_config(args: argparse.Namespace) -> argparse.Namespace:
    """Overlay a YAML config file onto the parsed arguments.

    Args:
        args: Parsed arguments; mutated in place.

    Returns:
        The same namespace.
    """
    if not args.config:
        return args

    import yaml

    with open(args.config) as f:
        cfg = yaml.safe_load(f) or {}

    for key in _CONFIG_KEYS:
        if key in cfg:
            setattr(args, key, cfg[key])
    return args


def _iter_slab_files(args: argparse.Namespace):
    """Yield the slab files selected by ``--slab-file`` or ``--slab-dir``.

    Args:
        args: Parsed arguments.

    Yields:
        Paths to slab files, sorted when a directory was given.

    Raises:
        ValueError: If neither ``--slab-file`` nor ``--slab-dir`` was given.
    """
    from pathlib import Path

    if args.slab_file:
        yield Path(args.slab_file)
    elif args.slab_dir:
        for path in sorted(Path(args.slab_dir).rglob(args.pattern)):
            if path.is_file():
                yield path
    else:
        raise ValueError("Provide --slab-file or --slab-dir (on the command line or in --config).")


def run(args: argparse.Namespace) -> int:
    """Generate and write adsorbate placements for every selected slab.

    Args:
        args: Parsed arguments from :func:`add_arguments`.

    Returns:
        Process exit status.
    """
    import random
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import yaml
    from ase.io import read, write
    from fairchem.data.oc.core.adsorbate import Adsorbate
    from fairchem.data.oc.core.adsorbate_slab_config import AdsorbateSlabConfig
    from fairchem.data.oc.core.multi_adsorbate_slab_config import MultipleAdsorbateSlabConfig

    from perovml.core.placement import PbAdsorbateSlabConfig
    from perovml.utils.structure import build_adsorbate, build_oc_slab_from_atoms

    args = _merge_config(args)

    if isinstance(args.seed, int):
        random.seed(args.seed)
        np.random.seed(args.seed)

    run_dir = Path(args.out_root) / datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"[Info] Run directory: {run_dir}")

    ads = build_adsorbate(
        molecule_file=args.molecule_file,
        adsorbate=args.adsorbate,
        binding_index=args.binding_index,
        run_dir=run_dir,
    )

    # Save the resolved options for reproducibility (after any binding-index guess)
    resolved_cfg = {key: getattr(args, key, None) for key in _CONFIG_KEYS}
    resolved_cfg["out_root"] = str(args.out_root)
    resolved_cfg["binding_index"] = int(ads.binding_indices[0]) if len(ads.binding_indices) else None
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(resolved_cfg, f, sort_keys=True)

    for slab_path in _iter_slab_files(args):
        try:
            slab_atoms = read(slab_path)
        except (OSError, ValueError) as exc:
            print(f"[Warn] Failed to read slab file {slab_path}: {exc}")
            continue

        slab = build_oc_slab_from_atoms(slab_atoms, min_ab=args.min_ab, surface_fraction=args.surface_fraction)

        outdir = run_dir / slab_path.stem
        outdir.mkdir(parents=True, exist_ok=True)
        write(str(outdir / "slab.vasp"), slab.atoms, format="vasp", direct=True)

        if args.which == "single":
            if args.mode in ("Pb_uniform_sample", "Pb_heuristic_sample"):
                sampler = PbAdsorbateSlabConfig(
                    slab=slab,
                    adsorbate=ads,
                    num_orientations=args.pb_num_orientations,
                    cone_angle_deg=args.pb_cone_deg,
                    interstitial_gap=args.interstitial_gap,
                    mode=args.mode,
                    # The sampler has its own generator, so --seed has to reach it
                    # directly; seeding the global one no longer affects it.
                    rng=args.seed if isinstance(args.seed, int) else None,
                )
            else:
                sampler = AdsorbateSlabConfig(
                    slab,
                    ads,
                    num_sites=args.num_sites,
                    num_augmentations_per_site=args.num_augmentations,
                    mode=args.mode,
                )
            prefix = "adslab_single"
        else:
            ads_list = [
                Adsorbate(
                    adsorbate_atoms=ads.atoms.copy(),
                    adsorbate_binding_indices=list(ads.binding_indices),
                )
                for _ in range(max(1, args.copies))
            ]
            sampler = MultipleAdsorbateSlabConfig(slab, ads_list, num_configurations=args.num_configurations)
            prefix = "adslab_multi"

        for i, atoms in enumerate(sampler.atoms_list):
            write(str(outdir / f"{prefix}_{i}.vasp"), atoms, format="vasp", direct=True)
        print(f"[Done] {slab_path.stem}: wrote {len(sampler.atoms_list)} placements to {outdir}")

    return 0
