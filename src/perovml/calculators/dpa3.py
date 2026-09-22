"""ASE calculator wrapper for the DPA-3 ``Omat24`` multi-task head.

This wraps the DeePMD-kit ``DP`` calculator and pins the multi-task head to
``Omat24`` so that it can be dropped into the perovml relaxation pipeline
(LBFGS / BFGS / FIRE).

Examples:
    >>> from perovml.calculators import DPA3Omat24Calculator
    >>> calc = DPA3Omat24Calculator(model_path="DPA-3.1-3M.pt")  # doctest: +SKIP

Notes:
    DeePMD-kit is an optional dependency; install it with
    ``pip install "perovml[dpa3]"``. The model file is not shipped with this
    repository -- see ``models/README.md`` for how to obtain it.
"""

from __future__ import annotations

from typing import Any

from ase.atoms import Atoms
from ase.calculators.calculator import Calculator


class DPA3Omat24Calculator(Calculator):
    """ASE calculator delegating to a DeePMD-kit DPA-3 model.

    Attributes:
        implemented_properties: Properties this calculator can return.
    """

    implemented_properties = ["energy", "forces"]

    def __init__(self, model_path: str, head: str = "Omat24", **kwargs: Any) -> None:
        """Initialize the calculator from a DeePMD-kit model file.

        Args:
            model_path: Path to the DPA-3 model file (e.g. ``DPA-3.1-3M.pt``).
            head: Multi-task head to select. Defaults to ``"Omat24"``.
            **kwargs: Reserved for forward compatibility; currently unused.

        Raises:
            ImportError: If DeePMD-kit is not installed.
        """
        super().__init__()
        try:
            from deepmd.calculator import DP as DPCalculator  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - runtime import guard
            raise ImportError(
                "deepmd-kit is required for DPA3Omat24Calculator. Install it with `pip install deepmd-kit`."
            ) from exc

        self._dp = DPCalculator(model_path, head=head)

    def calculate(
        self,
        atoms: Atoms | None = None,
        properties: tuple[str, ...] = ("energy",),
        system_changes: list[str] | None = None,
    ) -> None:
        """Delegate the calculation to the underlying DeePMD-kit calculator.

        Args:
            atoms: Structure to evaluate. Defaults to the attached structure.
            properties: Properties to compute.
            system_changes: ASE change list describing what differs from the
                previous call.
        """
        self._dp.calculate(atoms=atoms, properties=properties, system_changes=system_changes)
        self.results = dict(self._dp.results)
