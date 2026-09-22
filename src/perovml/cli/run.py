"""Sample adsorbate placements on a slab and relax them with an MLP calculator.

Everything is driven by a YAML config; see ``docs/parameters.md`` for the meaning
of each key. Results land in a timestamped directory under ``--out-root``: the
relaxed references, the generated and relaxed candidate structures, and
``report.csv`` / ``report.json``.

This subcommand handles one slab and one molecule, so its config names them with
``slab_file`` and ``molecule_file`` (or ``adsorbate``, a FAIRChem database key).
The configs under ``examples/configs/`` drive the batch scripts in ``scripts/``
instead, which sweep a whole directory of molecules and spell those keys ``slab``
and ``molecules_dir``; they are not interchangeable.

Examples:
    perovml run -c my_single_molecule.yaml --out-root outputs
"""

from __future__ import annotations

import argparse

DEFAULT_OUT_ROOT = "outputs"


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``perovml run`` arguments.

    Args:
        parser: Subcommand parser to populate.
    """
    parser.add_argument("-c", "--config", required=True, help="YAML config file path")
    parser.add_argument(
        "--out-root",
        default=DEFAULT_OUT_ROOT,
        help=f"Directory the timestamped run folder is created in (default: {DEFAULT_OUT_ROOT})",
    )


def run(args: argparse.Namespace) -> int:
    """Execute the config-driven sampling and relaxation pipeline.

    Args:
        args: Parsed arguments from :func:`add_arguments`.

    Returns:
        Process exit status.

    Raises:
        KeyError: If the config does not name a slab with ``slab_file``.
        ValueError: If the calculator or the optimizer name is not recognised.
    """
    import csv
    import json
    from datetime import datetime
    from pathlib import Path

    import numpy as np
    import yaml
    from ase.io import read, write

    from perovml.calculators.factory import build_calculator, build_optimizer_cls, resolve_calculator_name
    from perovml.core.recipes import run_perovml
    from perovml.utils.structure import build_adsorbate, build_oc_slab_from_atoms

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    if "slab_file" not in cfg:
        raise KeyError(
            "the config must set `slab_file`. A config written for the batch scripts in "
            "scripts/ (which uses `slab` and `molecules_dir`) will not drive `perovml run`; "
            "see docs/parameters.md."
        )

    # Check the names the config picked before anything is written, so that a typo costs
    # nothing and leaves nothing behind.
    resolve_calculator_name(cfg.get("calculator"))
    optimizer_cls = build_optimizer_cls(cfg.get("optimizer_cls", "LBFGS"))

    # timestamped run dir
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(args.out_root) / f"perovml_{ts}"
    run_dir.mkdir(parents=True, exist_ok=True)

    # save config snapshot
    with open(run_dir / "config.yaml", "w") as f:
        yaml.safe_dump(cfg, f, sort_keys=True)

    slab_atoms = read(cfg["slab_file"])
    slab = build_oc_slab_from_atoms(slab_atoms, min_ab=cfg.get("min_ab", 8.0))

    ads = build_adsorbate(
        molecule_file=cfg.get("molecule_file"),
        adsorbate=cfg.get("adsorbate"),
        binding_index=cfg.get("binding_index"),
        run_dir=run_dir,
    )

    calc, calculator_provenance = build_calculator(cfg)

    # Every candidate set is reproducible from this seed, which is recorded below.
    seed = cfg.get("seed")
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % (2**63))

    # Optional precomputed references; supplying one skips the matching relaxation.
    precomputed_slab_atoms = None
    precomputed_slab_energy = cfg.get("relaxed_slab_energy")
    if cfg.get("relaxed_slab_file"):
        precomputed_slab_atoms = read(cfg["relaxed_slab_file"])

    precomputed_gas_atoms = None
    precomputed_gas_energy = cfg.get("relaxed_adsorbate_energy")
    if cfg.get("relaxed_adsorbate_file"):
        precomputed_gas_atoms = read(cfg["relaxed_adsorbate_file"])

    outputs = run_perovml(
        slab=slab,
        adsorbate=ads if cfg.get("molecule_file") else cfg.get("adsorbate"),
        calculator=calc,
        optimizer_cls=optimizer_cls,
        fmax=cfg.get("fmax", 0.02),
        steps=cfg.get("steps", 100),
        generator=cfg.get("generator", "Pb_uniform_sample"),
        # These three match perov_adslab_generator's own signature defaults, so that the
        # documented default is the effective one no matter which entry point is used.
        num_sites=cfg.get("num_sites", 100),
        num_augmentations=cfg.get("num_augmentations", 1),
        num_configurations=cfg.get("num_configurations", 1),
        pb_num_orientations=cfg.get("pb_num", 20),
        pb_cone_deg=cfg.get("pb_cone", 20.0),
        interstitial_gap=cfg.get("interstitial_gap", 0.1),
        rng=seed,
        mode=cfg.get("mode", "random"),
        # Precomputed references
        precomputed_slab_atoms=precomputed_slab_atoms,
        precomputed_slab_energy=precomputed_slab_energy,
        precomputed_gas_atoms=precomputed_gas_atoms,
        precomputed_gas_energy=precomputed_gas_energy,
    )

    # save relaxed references
    refs = outputs.get("references", {})
    slab_relaxed = refs.get("slab_atoms")
    gas_relaxed = refs.get("gas_adsorbate_atoms")
    if slab_relaxed is not None:
        write(str(run_dir / "slab_relaxed.vasp"), slab_relaxed, format="vasp", direct=True)
    if gas_relaxed is not None:
        write(str(run_dir / "adsorbate_relaxed.vasp"), gas_relaxed, format="vasp", direct=True)

    # write all candidates and a report
    rows = []
    init_dir = run_dir / "initials"
    cand_dir = run_dir / "candidates"
    init_dir.mkdir(exist_ok=True)
    cand_dir.mkdir(exist_ok=True)
    for i, r in enumerate(outputs["adslabs"]):
        atoms = r["atoms"]
        energy = r["results"]["energy"]
        eads = r["results"].get("adsorption_energy")
        anomalies = r["results"].get("anomalies", [])
        # save final
        write(str(cand_dir / f"cand_{i}.vasp"), atoms, format="vasp", direct=True)
        # save initial (generated) if available from relax_job input
        init_atoms = (r.get("input_atoms") or {}).get("atoms")
        if init_atoms is not None:
            write(str(init_dir / f"cand_{i}.vasp"), init_atoms, format="vasp", direct=True)
        rows.append(
            {
                "idx": i,
                "energy": float(energy),
                "adsorption_energy": (None if eads is None else float(eads)),
                "anomalies": ";".join(anomalies) if anomalies else "",
            }
        )

    # What produced these numbers, and what it would take to reproduce them.
    with open(run_dir / "provenance.json", "w") as f:
        json.dump(
            {
                "calculator": calculator_provenance,
                "optimizer_cls": optimizer_cls.__name__,
                "fmax": cfg.get("fmax", 0.02),
                "steps": cfg.get("steps", 100),
                "generator": cfg.get("generator", "Pb_uniform_sample"),
                "seed": seed,
            },
            f,
            indent=2,
        )

    # CSV report
    with open(run_dir / "report.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["idx", "energy", "adsorption_energy", "anomalies"])
        writer.writeheader()
        writer.writerows(rows)
    # JSON full
    with open(run_dir / "report.json", "w") as f:
        json.dump(rows, f, indent=2)

    # save top for convenience
    if outputs["adslabs"]:
        write(str(run_dir / "top.vasp"), outputs["adslabs"][0]["atoms"], format="vasp", direct=True)

    print(f"[Done] perovml run wrote {len(rows)} candidates to {run_dir}")
    print(f"[Info] calculator: {calculator_provenance['class']}, seed: {seed} (see provenance.json)")
    return 0
