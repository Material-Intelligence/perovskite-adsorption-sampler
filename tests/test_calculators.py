"""Tests for the ASE calculators bundled with perovml.

:class:`~perovml.calculators.mock.MockCalculator` is always exercised. The DPA-3
calculator needs a model file that this repository does not ship, so its test is
skipped unless ``DPA3_MODEL_PATH`` points at one (see ``models/README.md``).
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from perovml.calculators.mock import MockCalculator

#: Environment variable holding the path to a DPA-3 model file.
DPA3_MODEL_PATH_ENV = "DPA3_MODEL_PATH"

_DPA3_MODEL_PATH = os.environ.get(DPA3_MODEL_PATH_ENV, "")

requires_dpa3_model = pytest.mark.skipif(
    not _DPA3_MODEL_PATH or not Path(_DPA3_MODEL_PATH).is_file(),
    reason=f"set {DPA3_MODEL_PATH_ENV} to a DPA-3 model file to run this test",
)


@pytest.fixture
def probe_atoms():
    """A three-atom cell, built with pymatgen and handed to ASE."""
    lattice = Lattice.from_parameters(10.0, 10.0, 10.0, 90.0, 90.0, 90.0)
    structure = Structure(lattice, ["C", "O", "O"], [[0.2, 0.3, 0.4], [0.3, 0.3, 0.4], [0.2, 0.4, 0.4]])
    return AseAtomsAdaptor.get_atoms(structure)


class TestMockCalculator:
    """The mock calculator is a quadratic well; its numbers are meaningless."""

    def test_returns_energy_and_forces(self, probe_atoms):
        probe_atoms.calc = MockCalculator()
        energy = probe_atoms.get_potential_energy()
        forces = probe_atoms.get_forces()

        assert np.isfinite(energy)
        assert forces.shape == (len(probe_atoms), 3)
        assert np.isfinite(forces).all()

    def test_forces_match_the_energy_gradient(self, probe_atoms):
        probe_atoms.calc = MockCalculator()
        analytic = probe_atoms.get_forces()

        delta = 1e-4
        numeric = np.zeros_like(analytic)
        for atom in range(len(probe_atoms)):
            for axis in range(3):
                for sign in (+1, -1):
                    shifted = probe_atoms.copy()
                    positions = shifted.get_positions()
                    positions[atom, axis] += sign * delta
                    shifted.set_positions(positions)
                    shifted.calc = MockCalculator()
                    numeric[atom, axis] -= sign * shifted.get_potential_energy() / (2 * delta)

        assert np.allclose(analytic, numeric, atol=1e-6)

    def test_forces_point_towards_the_cell_origin(self, probe_atoms):
        probe_atoms.calc = MockCalculator()
        forces = probe_atoms.get_forces()
        assert np.all(np.sign(forces) == -np.sign(probe_atoms.get_positions()))


@requires_dpa3_model
def test_dpa3_calculator_evaluates_a_small_cell(probe_atoms):
    """Smoke test for the DPA-3 wrapper; needs a model file and deepmd-kit."""
    from perovml.calculators.dpa3 import DPA3Omat24Calculator

    probe_atoms.calc = DPA3Omat24Calculator(model_path=_DPA3_MODEL_PATH)
    energy = probe_atoms.get_potential_energy()
    forces = probe_atoms.get_forces()

    assert np.isfinite(energy)
    assert forces.shape == (len(probe_atoms), 3)
