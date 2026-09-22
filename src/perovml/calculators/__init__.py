"""Calculator wrappers for various ML potentials, and the factory that selects one."""

from perovml.calculators.factory import (
    CALCULATOR_ALIASES,
    build_calculator,
    build_optimizer_cls,
    resolve_calculator_name,
)
from perovml.calculators.mock import MockCalculator

__all__ = [
    "CALCULATOR_ALIASES",
    "MockCalculator",
    "build_calculator",
    "build_optimizer_cls",
    "resolve_calculator_name",
]

# DPA-3 is an optional extra (`pip install "perovml[dpa3]"`); expose it only when
# DeePMD-kit is importable.
try:
    from perovml.calculators.dpa3 import DPA3Omat24Calculator  # noqa: F401

    __all__.append("DPA3Omat24Calculator")
except ImportError:  # pragma: no cover - runtime import guard
    pass
