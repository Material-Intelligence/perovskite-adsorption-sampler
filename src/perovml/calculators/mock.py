"""A dependency-free ASE calculator used to exercise the pipeline end to end."""

from __future__ import annotations

import numpy as np
from ase.atoms import Atoms
from ase.calculators.calculator import Calculator


class MockCalculator(Calculator):
    """Toy calculator whose forces pull every atom towards the cell origin.

    It needs no machine-learning model and no GPU, so it is the calculator the
    examples and tests use to check that sampling, relaxation and bookkeeping run.
    The energies it returns are meaningless.

    Attributes:
        implemented_properties: Properties this calculator can return.
    """

    implemented_properties = ["energy", "forces"]

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: tuple[str, ...] = ("energy",),
        system_changes: list[str] | None = None,
    ) -> None:
        """Evaluate the quadratic well around the cell origin.

        Args:
            atoms: Structure to evaluate. Defaults to the attached structure.
            properties: Properties to compute.
            system_changes: ASE change list describing what differs from the
                previous call.
        """
        Calculator.calculate(self, atoms, properties, system_changes)
        # Simple quadratic well around the origin (fakes a soft relaxation)
        pos = self.atoms.get_positions()
        forces = -0.01 * pos
        energy = float(0.5 * 0.01 * np.sum(pos**2))
        self.results = {"energy": energy, "forces": forces}
