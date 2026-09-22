#!/usr/bin/env python3
"""Flag adsorption structures that came out of a run looking broken.

The checks are deliberately crude geometric sanity tests, meant to catch the
obvious failures before anyone spends DFT time on them:

  1. atoms closer than 0.8 A, i.e. overlapping;
  2. an adsorbate whose own atoms are far apart, i.e. dissociated or scattered;
  3. an adsorbate that is embedded in the slab or has drifted away from it;
  4. an adsorption energy that is implausibly negative, or positive.

Examples:
    python scripts/check_structures.py outputs/parallel/<timestamp>
    python scripts/check_structures.py outputs/parallel/<timestamp> --threshold -5.0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
from ase.io import read
from scipy.spatial.distance import cdist, pdist

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

MIN_ATOM_DISTANCE = 0.8
"""Below this interatomic distance, in A, the atoms are treated as overlapping."""

MAX_ADSORBATE_SPAN = 15.0
"""Above this largest interatomic distance, in A, the adsorbate is treated as dispersed."""

MAX_ADSORBATE_Z_SPAN = 12.0
"""Above this z extent, in A, the adsorbate is treated as stretched across the cell."""

MIN_ADSORBATE_SLAB_DISTANCE = 1.0
"""Below this adsorbate-to-slab distance, in A, the molecule is treated as embedded."""

MAX_ADSORBATE_SLAB_DISTANCE = 6.0
"""Above this adsorbate-to-slab distance, in A, the molecule is treated as desorbed."""

SURFACE_ELEMENTS = frozenset({"Pb", "I"})
"""Elements that belong to the slab; finding one in the adsorbate means the split went wrong."""


def check_structure(vasp_file: str | Path, slab_z_threshold: float = 0.45) -> dict[str, Any]:
    """Run the geometric checks on one relaxed adslab.

    The adsorbate is separated from the slab by fractional z alone, which is
    enough for the slab geometries this repository ships.

    Args:
        vasp_file: Path to the relaxed structure.
        slab_z_threshold: Fractional z above which an atom counts as adsorbate.

    Returns:
        A record with the atom counts, the measured distances and an ``anomalies``
        list, or ``{"error": ...}`` when the file could not be read.
    """
    try:
        atoms = read(vasp_file)
    except Exception as exc:  # noqa: BLE001 - an unreadable file is a result, not a crash
        return {"error": str(exc)}

    positions = atoms.get_positions()
    z_scaled = atoms.get_scaled_positions()[:, 2]
    symbols = np.array(atoms.get_chemical_symbols())

    slab_mask = z_scaled < slab_z_threshold
    ads_mask = z_scaled >= slab_z_threshold

    result: dict[str, Any] = {
        "n_atoms": len(atoms),
        "n_slab": int(slab_mask.sum()),
        "n_adsorbate": int(ads_mask.sum()),
        "anomalies": [],
    }

    if ads_mask.sum() == 0:
        result["anomalies"].append("no_adsorbate_found")
        return result

    ads_positions = positions[ads_mask]
    ads_symbols = symbols[ads_mask]
    slab_positions = positions[slab_mask]

    # 1. Interatomic distances inside the adsorbate.
    if len(ads_positions) > 1:
        ads_dists = pdist(ads_positions)
        min_ads_dist = float(ads_dists.min())
        max_ads_dist = float(ads_dists.max())
        result["min_ads_dist"] = min_ads_dist
        result["max_ads_dist"] = max_ads_dist

        if min_ads_dist < MIN_ATOM_DISTANCE:
            result["anomalies"].append(f"atom_overlap:{min_ads_dist:.3f}A")

        if max_ads_dist > MAX_ADSORBATE_SPAN:
            result["anomalies"].append(f"molecule_dispersed:{max_ads_dist:.2f}A")

    # 2. How far the adsorbate reaches along z.
    z_range = ads_positions[:, 2].max() - ads_positions[:, 2].min()
    result["ads_z_range"] = float(z_range)
    if z_range > MAX_ADSORBATE_Z_SPAN:
        result["anomalies"].append(f"large_z_span:{z_range:.2f}A")

    # 3. Distance between the adsorbate and the slab.
    if len(slab_positions) > 0:
        min_cross_dist = float(cdist(ads_positions, slab_positions).min())
        result["min_ads_slab_dist"] = min_cross_dist

        if min_cross_dist < MIN_ADSORBATE_SLAB_DISTANCE:
            result["anomalies"].append(f"embedded_in_slab:{min_cross_dist:.3f}A")
        elif min_cross_dist > MAX_ADSORBATE_SLAB_DISTANCE:
            result["anomalies"].append(f"far_from_slab:{min_cross_dist:.2f}A")

    # 4. Slab elements that ended up on the adsorbate side of the cut.
    ads_elements = set(ads_symbols)
    contamination = ads_elements & SURFACE_ELEMENTS
    if contamination:
        result["anomalies"].append(f"surface_atoms_in_adsorbate:{sorted(contamination)}")

    result["ads_elements"] = sorted(ads_elements)
    result["is_anomalous"] = len(result["anomalies"]) > 0

    return result


def check_all_structures(
    output_dir: str | Path,
    energy_threshold: float = -5.0,
    check_rank: int = 1,
) -> pd.DataFrame:
    """Check every molecule in a run directory.

    Args:
        output_dir: Run directory holding one sub-directory per molecule.
        energy_threshold: Adsorption energies below this, in eV, are suspicious.
        check_rank: Which ranked structure to check; 1 is the best one.

    Returns:
        One row per molecule, with the energies, the measured distances and the
        anomalies that were found.
    """
    import pandas as pd

    output_path = Path(output_dir)
    records = []

    for mol_dir in sorted(output_path.iterdir()):
        if not mol_dir.is_dir():
            continue

        summary_file = mol_dir / "summary.json"
        structure_file = mol_dir / f"rank_{check_rank}.vasp"

        if not summary_file.exists():
            continue

        with open(summary_file) as file:
            summary = json.load(file)

        candidates = summary.get("candidates", [])
        if not candidates:
            continue

        best = candidates[0]
        e_ads = best.get("adsorption_energy")

        record: dict[str, Any] = {
            "sid": mol_dir.name,
            "energy": best.get("energy"),
            "adsorption_energy": e_ads,
        }

        energy_anomaly = False
        if e_ads is not None:
            if e_ads < energy_threshold:
                energy_anomaly = True
                record["energy_anomaly"] = f"too_low:{e_ads:.2f}eV"
            elif e_ads > 0:
                energy_anomaly = True
                record["energy_anomaly"] = f"positive:{e_ads:.2f}eV"

        if structure_file.exists():
            struct_check = check_structure(structure_file)
            record.update(
                {
                    "n_adsorbate": struct_check.get("n_adsorbate"),
                    "min_ads_dist": struct_check.get("min_ads_dist"),
                    "max_ads_dist": struct_check.get("max_ads_dist"),
                    "ads_z_range": struct_check.get("ads_z_range"),
                    "min_ads_slab_dist": struct_check.get("min_ads_slab_dist"),
                    "structure_anomalies": "; ".join(struct_check.get("anomalies", [])),
                    "is_anomalous": struct_check.get("is_anomalous", False) or energy_anomaly,
                }
            )
        else:
            record["structure_anomalies"] = "file_not_found"
            record["is_anomalous"] = True

        records.append(record)

    return pd.DataFrame(records)


def parse_args() -> argparse.Namespace:
    """Parse the command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Flag broken adsorption structures")
    parser.add_argument("output_dir", help="Run directory to check")
    parser.add_argument(
        "--threshold",
        type=float,
        default=-5.0,
        help="Adsorption energies below this, in eV, are treated as suspicious (default: -5.0)",
    )
    parser.add_argument("--rank", type=int, default=1, help="Which ranked structure to check (default: 1, the best)")
    parser.add_argument("--save", help="Write the report to this CSV file")
    return parser.parse_args()


def main() -> None:
    """Check a run directory and report the structures that look wrong."""
    import pandas as pd

    args = parse_args()

    print("=" * 70)
    print("  Adsorption structure check")
    print("=" * 70)
    print(f"directory: {args.output_dir}")
    print(f"energy threshold: {args.threshold} eV")
    print()

    dataframe = check_all_structures(args.output_dir, args.threshold, args.rank)

    if len(dataframe) == 0:
        print("[Warn] no results found")
        return

    n_total = len(dataframe)
    n_anomalous = int(dataframe["is_anomalous"].sum())

    print(f"molecules: {n_total}")
    print(f"flagged:   {n_anomalous} ({100 * n_anomalous / n_total:.1f}%)")
    print()

    if n_anomalous > 0:
        print("-" * 70)
        print("Flagged molecules:")
        print("-" * 70)

        anomalous = dataframe[dataframe["is_anomalous"]].copy().sort_values("adsorption_energy")

        for _, row in anomalous.iterrows():
            print(f"\n[{row['sid']}]")
            print(f"  adsorption energy: {row['adsorption_energy']:.4f} eV")
            if pd.notna(row.get("energy_anomaly")):
                print(f"  energy anomaly:    {row['energy_anomaly']}")
            if row.get("structure_anomalies"):
                print(f"  structure anomaly: {row['structure_anomalies']}")
            if pd.notna(row.get("min_ads_dist")):
                print(f"  smallest adsorbate interatomic distance: {row['min_ads_dist']:.3f} A")
            if pd.notna(row.get("max_ads_dist")):
                print(f"  largest adsorbate interatomic distance:  {row['max_ads_dist']:.2f} A")

    save_path = Path(args.save) if args.save else Path(args.output_dir) / "structure_check.csv"
    dataframe.to_csv(save_path, index=False)
    print(f"\n[Info] report written to {save_path}")

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
