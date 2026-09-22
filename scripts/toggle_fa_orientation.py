#!/usr/bin/env python

"""
Toggle FA molecular orientations in a FAPbI3 3x3 slab POSCAR.

This script is tailored for data/slabs/FAPbI3_3x3_zcut017.vasp in this repo:
- Species order: C N H Pb I
- Counts:        18 36 90 18 54  (18 FA molecules = 3x3x2)
- Coordinates:   Direct

It identifies the 18 FA molecules via the C atoms, arranges them in a 3x3
grid for each of the two FA layers along c, and then flips the whole FA
molecule orientation along the b lattice direction according to the pattern
below, by reflecting all atoms of selected FA molecules with respect to a
plane normal to b passing through that FA's center (along b). This keeps the
FA center fixed and preserves all internal bond lengths and angles; only the
"direction" along b is reversed.

Layer 0 (lower z, first FA layer):
    1 0 1
    0 1 0
    1 0 1

Layer 1 (higher z, second FA layer):
    0 1 0
    1 0 1
    0 1 0

Here 1 means "keep current +b orientation", 0 means "flip to -b" by mirroring
the entire FA (all its atoms) across a plane normal to b.

Usage:
    python scripts/toggle_fa_orientation.py \
        --input data/slabs/FAPbI3_3x3_zcut017.vasp \
        --output outputs/FAPbI3_3x3_zcut017_altFA.vasp
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import List, Tuple


def read_poscar(path: Path):
    lines = path.read_text().splitlines()
    if len(lines) < 9:
        raise ValueError("POSCAR file too short.")

    comment = lines[0]
    scale = float(lines[1].split()[0])
    lattice = []
    for i in range(2, 5):
        lattice.append([scale * float(x) for x in lines[i].split()[:3]])

    # species and counts
    species_line = lines[5].split()
    counts = [int(x) for x in lines[6].split()]
    total_atoms = sum(counts)

    coord_start = 7
    coord_type_line = lines[coord_start].strip().lower()
    selective = False
    if coord_type_line.startswith("s"):
        selective = True
        coord_start += 1
        coord_type_line = lines[coord_start].strip().lower()

    if not (coord_type_line.startswith("d") or coord_type_line.startswith("c")):
        raise ValueError("Unknown coordinate type (expect Direct or Cartesian).")

    coord_type = "direct" if coord_type_line.startswith("d") else "cartesian"
    coords: List[List[float]] = []
    flags: List[Tuple[str, str, str]] | None = [] if selective else None

    for i in range(total_atoms):
        parts = lines[coord_start + 1 + i].split()
        if len(parts) < 3:
            raise ValueError(f"Not enough coordinates on line {coord_start + 2 + i}.")
        x, y, z = map(float, parts[:3])
        coords.append([x, y, z])
        if selective:
            # keep whatever flags are present; pad to 3 if shorter
            fl = parts[3:6]
            while len(fl) < 3:
                fl.append("T")
            flags.append(tuple(fl[:3]))

    return {
        "comment": comment,
        "scale": scale,
        "lattice": lattice,
        "species": species_line,
        "counts": counts,
        "coord_type": coord_type,
        "selective": selective,
        "coords": coords,
        "flags": flags,
        "header_lines": lines[: coord_start + 1],
    }


def frac_to_cart(frac: List[float], lattice) -> List[float]:
    ax, ay, az = lattice[0]
    bx, by, bz = lattice[1]
    cx, cy, cz = lattice[2]
    u, v, w = frac
    return [
        u * ax + v * bx + w * cx,
        u * ay + v * by + w * cy,
        u * az + v * bz + w * cz,
    ]


def cart_to_frac(cart: List[float], lattice) -> List[float]:
    # invert 3x3 lattice matrix
    ax, ay, az = lattice[0]
    bx, by, bz = lattice[1]
    cx, cy, cz = lattice[2]
    det = ax * (by * cz - bz * cy) - ay * (bx * cz - bz * cx) + az * (bx * cy - by * cx)
    if abs(det) < 1e-12:
        raise ValueError("Lattice matrix is singular.")

    inv = [[0.0] * 3 for _ in range(3)]
    inv[0][0] = (by * cz - bz * cy) / det
    inv[0][1] = (az * cy - ay * cz) / det
    inv[0][2] = (ay * bz - az * by) / det
    inv[1][0] = (bz * cx - bx * cz) / det
    inv[1][1] = (ax * cz - az * cx) / det
    inv[1][2] = (az * bx - ax * bz) / det
    inv[2][0] = (bx * cy - by * cx) / det
    inv[2][1] = (ay * cx - ax * cy) / det
    inv[2][2] = (ax * by - ay * bx) / det

    x, y, z = cart
    u = inv[0][0] * x + inv[0][1] * y + inv[0][2] * z
    v = inv[1][0] * x + inv[1][1] * y + inv[1][2] * z
    w = inv[2][0] * x + inv[2][1] * y + inv[2][2] * z
    return [u, v, w]


def wrap_frac(frac: List[float]) -> List[float]:
    return [f - math.floor(f) for f in frac]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Input POSCAR (C N H Pb I order, 18/36/90 C/N/H).")
    parser.add_argument("--output", required=True, help="Output POSCAR path.")
    return parser.parse_args()


def main() -> None:
    """Flip alternating FA molecules and write the modified POSCAR."""
    args = parse_args()
    in_path = Path(args.input)
    out_path = Path(args.output)

    data = read_poscar(in_path)
    species = data["species"]
    counts = data["counts"]
    coords = data["coords"]
    lattice = data["lattice"]

    if species[:3] != ["C", "N", "H"]:
        raise ValueError(f"Unexpected species order {species}; this script expects C N H Pb I.")
    if len(counts) < 3:
        raise ValueError("Not enough species counts; expect at least C, N, H.")

    nC, nN, nH = counts[0], counts[1], counts[2]
    if nC != 18 or nN != 36 or nH != 90:
        raise ValueError(f"Unexpected C/N/H counts {nC}/{nN}/{nH}; " "this script is tailored for 18/36/90 (3x3x2 FA).")

    total = sum(counts)
    if len(coords) != total:
        raise ValueError("Coordinate count does not match species counts.")

    # Slices for species in coords list (Direct)
    c_start = 0
    n_start = c_start + nC
    h_start = n_start + nN

    C_frac = coords[c_start : c_start + nC]
    # build Cartesian coordinates for grouping by distance
    cart_coords = [frac_to_cart(f, lattice) for f in coords]

    def dist2(i: int, j: int) -> float:
        xi, yi, zi = cart_coords[i]
        xj, yj, zj = cart_coords[j]
        dx = xi - xj
        dy = yi - yj
        dz = zi - zj
        return dx * dx + dy * dy + dz * dz

    # Assign each N to its nearest C to identify FA molecules
    n_to_c: dict[int, int] = {}
    for n_idx in range(n_start, n_start + nN):
        best_c = None
        best_d2 = float("inf")
        for c_idx in range(c_start, c_start + nC):
            d2 = dist2(n_idx, c_idx)
            if d2 < best_d2:
                best_d2 = d2
                best_c = c_idx
        if best_c is None:
            raise RuntimeError(f"Failed to find nearest C for N index {n_idx}.")
        n_to_c[n_idx] = best_c

    # Initialize FA groups: one per C
    fa_atoms: dict[int, List[int]] = {c_idx: [c_idx] for c_idx in range(c_start, c_start + nC)}

    # Add N atoms to their FA
    for n_idx, c_idx in n_to_c.items():
        fa_atoms[c_idx].append(n_idx)

    # Assign each H to closest C or N, then map to its FA (C)
    for h_idx in range(h_start, h_start + nH):
        best_idx = None
        best_d2 = float("inf")

        # compare with all C and N
        for idx in list(range(c_start, c_start + nC)) + list(range(n_start, n_start + nN)):
            d2 = dist2(h_idx, idx)
            if d2 < best_d2:
                best_d2 = d2
                best_idx = idx

        if best_idx is None:
            raise RuntimeError(f"Failed to find nearest C/N for H index {h_idx}.")

        if best_idx < n_start:
            c_idx = best_idx  # nearest is a C directly
        else:
            c_idx = n_to_c[best_idx]  # nearest is an N; map via its C

        fa_atoms[c_idx].append(h_idx)

    # Optionally sanity check: each FA should have 1C + 2N + 5H = 8 atoms
    for c_idx, atoms in fa_atoms.items():
        if len(atoms) != 8:
            raise RuntimeError(f"FA associated with C index {c_idx} has {len(atoms)} atoms, expected 8.")

    # Arrange 18 C atoms into two 3x3 layers along c
    c_indices = list(range(nC))
    c_indices_sorted_by_z = sorted(c_indices, key=lambda i: C_frac[i][2])
    bottom_layer = c_indices_sorted_by_z[:9]
    top_layer = c_indices_sorted_by_z[9:]

    def order_layer(indices: List[int]) -> List[int]:
        # sort by y, then within each row by x, to get 3x3 grid
        sorted_by_y = sorted(indices, key=lambda i: C_frac[i][1])
        ordered: List[int] = []
        for row in range(3):
            row_indices = sorted_by_y[row * 3 : (row + 1) * 3]
            row_indices = sorted(row_indices, key=lambda i: C_frac[i][0])
            ordered.extend(row_indices)
        return ordered

    bottom_ordered = order_layer(bottom_layer)
    top_ordered = order_layer(top_layer)

    # Desired orientation patterns (1 = keep, 0 = flip whole FA)
    pattern_bottom = [
        [1, 0, 1],
        [0, 1, 0],
        [1, 0, 1],
    ]
    pattern_top = [
        [0, 1, 0],
        [1, 0, 1],
        [0, 1, 0],
    ]

    def flip_fa_along_b(c_idx: int) -> None:
        """Reflect the whole FA group associated with C index c_idx along b.

        Implemented in Direct coordinates: x,z unchanged, y -> 2*y0 - y,
        where y0 is the average y of atoms in this FA. This keeps the FA
        center along b fixed and mirrors all atoms along b.
        """
        atoms = fa_atoms[c_idx]
        if not atoms:
            return
        y0 = sum(coords[i][1] for i in atoms) / float(len(atoms))
        for i in atoms:
            u, v, w = coords[i]
            v_new = 2.0 * y0 - v
            # wrap v into [0,1)
            v_new = v_new - math.floor(v_new)
            coords[i][1] = v_new

    def apply_pattern(layer_indices: List[int], pattern: List[List[int]]) -> None:
        for idx_in_layer, c_idx in enumerate(layer_indices):
            row = idx_in_layer // 3
            col = idx_in_layer % 3
            keep = pattern[row][col]
            if keep == 1:
                continue  # leave this FA as is
            # flip entire FA around its center along b
            flip_fa_along_b(c_idx)

    apply_pattern(bottom_ordered, pattern_bottom)
    apply_pattern(top_ordered, pattern_top)

    # Write out new POSCAR
    out_lines: List[str] = []
    out_lines.append(data["comment"])
    out_lines.append(f"{data['scale']: .16f}".strip())
    for row in lattice:
        out_lines.append("  " + "  ".join(f"{x: .16f}" for x in row))
    out_lines.append("  " + "  ".join(species))
    out_lines.append("  " + "  ".join(str(c) for c in counts))
    if data["selective"]:
        out_lines.append("Selective dynamics")
    out_lines.append("Direct")

    if data["selective"] and data["flags"] is not None:
        flags = data["flags"]
        for i, (x, y, z) in enumerate(coords):
            fx, fy, fz = flags[i]
            out_lines.append(f"  {x: .16f}  {y: .16f}  {z: .16f}   {fx} {fy} {fz}")
    else:
        for x, y, z in coords:
            out_lines.append(f"  {x: .16f}  {y: .16f}  {z: .16f}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("\n".join(out_lines) + "\n")

    print(f"Wrote modified POSCAR with alternating FA orientations to: {out_path}")


if __name__ == "__main__":
    main()
