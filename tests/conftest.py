"""Shared fixtures for the perovml test suite.

Every fixture here is built from the structures shipped in ``data/``, so the
suite runs without a network connection, a machine-learning model or a GPU.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from ase.atoms import Atoms
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data"

#: Pb-I terminated FAPbI3 slab, 216 atoms.
SLAB_FILE = DATA_DIR / "slabs" / "FAPbI3_3x3_zcut017.vasp"

#: Small probe molecule, 10 atoms, one carbonyl oxygen as the binding atom.
MOLECULE_FILE = DATA_DIR / "molecules" / "Acetone.vasp"


def read_structure(path: str | Path) -> Atoms:
    """Read a structure file with pymatgen and hand it to ASE.

    Args:
        path: Path to any file :meth:`pymatgen.core.Structure.from_file` accepts.

    Returns:
        The structure as an ASE :class:`~ase.atoms.Atoms` object.
    """
    return AseAtomsAdaptor.get_atoms(Structure.from_file(str(path)))


@pytest.fixture(scope="session")
def slab_atoms() -> Atoms:
    """The raw slab structure, without surface tags or constraints."""
    return read_structure(SLAB_FILE)


@pytest.fixture(scope="session")
def slab(slab_atoms: Atoms):
    """The slab wrapped in a FAIRChem ``Slab``, tagged and constrained."""
    from perovml.utils.structure import build_oc_slab_from_atoms

    return build_oc_slab_from_atoms(slab_atoms.copy(), min_ab=8.0)


@pytest.fixture(scope="session")
def adsorbate():
    """The probe molecule wrapped in a FAIRChem ``Adsorbate``."""
    from perovml.utils.structure import build_adsorbate

    return build_adsorbate(molecule_file=str(MOLECULE_FILE))
