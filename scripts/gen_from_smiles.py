#!/usr/bin/env python3
"""Generate gas-phase molecule POSCARs from SMILES strings.

RDKit embeds a 3D conformer and relaxes it with UFF; the molecule is then centred
in a cubic box and written as a POSCAR. The box only has to be large enough that
the molecule does not see its periodic images.

The two hydrazide molecules shipped in ``data/molecules`` were produced this way::

    python scripts/gen_from_smiles.py --outdir data/molecules \\
        --molecule "SE=NNC(N)=O" --molecule "CBH=NNC(=O)NN"

Exact coordinates depend on the RDKit version, so regenerating them will not
reproduce the shipped files byte for byte.

RDKit is not a declared dependency of perovml; install it separately
(``pip install rdkit`` or ``conda install -c conda-forge rdkit``).
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from pymatgen.core import Lattice, Structure


@dataclass(frozen=True)
class MoleculeSpec:
    """A molecule to build.

    Attributes:
        name: Output file stem and POSCAR title.
        smiles: SMILES string.
    """

    name: str
    smiles: str

    @classmethod
    def from_arg(cls, text: str) -> MoleculeSpec:
        """Parse a ``NAME=SMILES`` command-line argument.

        Args:
            text: The raw argument value.

        Returns:
            The parsed spec.

        Raises:
            argparse.ArgumentTypeError: If the argument is not ``NAME=SMILES``.
        """
        name, sep, smiles = text.partition("=")
        if not sep or not name or not smiles:
            raise argparse.ArgumentTypeError(f"expected NAME=SMILES, got {text!r}")
        return cls(name=name, smiles=smiles)


def smiles_to_structure(spec: MoleculeSpec, box: float = 30.0, seed: int = 42) -> Structure:
    """Build a 3D structure from a SMILES string, centred in a cubic box.

    Args:
        spec: Molecule name and SMILES string.
        box: Edge length of the cubic cell, in angstrom.
        seed: Random seed for RDKit's ETKDG conformer generation.

    Returns:
        The molecule as a periodic ``Structure`` in a cubic cell.

    Raises:
        SystemExit: If RDKit is not installed.
        ValueError: If the SMILES string cannot be parsed or embedded.
    """
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
    except ImportError as exc:
        raise SystemExit("RDKit is required for this script. Install it with `pip install rdkit`.") from exc

    mol = Chem.MolFromSmiles(spec.smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {spec.smiles}")

    mol = Chem.AddHs(mol)

    # Generate a 3D conformer and run a quick force-field relaxation
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(mol, params) != 0:
        raise ValueError(f"RDKit could not embed a 3D conformer for {spec.name} ({spec.smiles})")
    AllChem.UFFOptimizeMolecule(mol)

    conformer = mol.GetConformer()
    species = [atom.GetSymbol() for atom in mol.GetAtoms()]
    positions = np.array(
        [list(conformer.GetAtomPosition(atom.GetIdx())) for atom in mol.GetAtoms()],
        dtype=float,
    )

    # Centre the molecule in the middle of the box
    positions -= positions.mean(axis=0)
    positions += box / 2.0

    return Structure(
        Lattice.cubic(box),
        species,
        positions,
        coords_are_cartesian=True,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--molecule",
        action="append",
        required=True,
        dest="molecules",
        metavar="NAME=SMILES",
        type=MoleculeSpec.from_arg,
        help="Molecule to build, as NAME=SMILES. Repeat for each molecule.",
    )
    parser.add_argument("--outdir", default=".", help="Directory for the generated POSCARs (default: .)")
    parser.add_argument("--box", type=float, default=30.0, help="Cubic cell edge length in angstrom (default: 30)")
    parser.add_argument("--seed", type=int, default=42, help="RDKit conformer seed (default: 42)")
    return parser.parse_args()


def main() -> int:
    """Build every requested molecule and write it as a POSCAR.

    Returns:
        Process exit status.
    """
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    for spec in args.molecules:
        structure = smiles_to_structure(spec, box=args.box, seed=args.seed)
        out_path = outdir / f"{spec.name}.vasp"
        structure.to(filename=str(out_path), fmt="poscar", comment=f"{spec.name} {structure.composition.formula}")
        print(f"Wrote {out_path} ({structure.composition.reduced_formula}, {len(structure)} atoms)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
