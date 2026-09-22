"""perovml: adsorbate placement samplers and an MLP-to-DFT workflow for perovskite surfaces."""

from __future__ import annotations

from typing import Any

__version__ = "0.1.0"

__all__ = [
    "MockCalculator",
    "PbAdsorbateSlabConfig",
    "__version__",
    "perov_adslab_generator",
    "perov_ml_pipeline",
    "relax_job",
    "run_perovml",
]

# Most of the public API pulls in FAIRChem, which is a heavy import. Resolve those
# names lazily (PEP 562) so that `import perovml`, `perovml --help` and the parts of
# the package that only need ASE or pymatgen stay usable without it.
_LAZY_IMPORTS: dict[str, str] = {
    "MockCalculator": "perovml.calculators.mock",
    "PbAdsorbateSlabConfig": "perovml.core.placement",
    "perov_adslab_generator": "perovml.core.recipes",
    "perov_ml_pipeline": "perovml.core.recipes",
    "relax_job": "perovml.core.recipes",
    "run_perovml": "perovml.core.recipes",
}


def __getattr__(name: str) -> Any:
    """Import a public symbol on first access.

    Args:
        name: Attribute being looked up on the ``perovml`` package.

    Returns:
        The requested object.

    Raises:
        AttributeError: If ``name`` is not part of the public API.
    """
    module_name = _LAZY_IMPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    from importlib import import_module

    value = getattr(import_module(module_name), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """List the public API, including the lazily imported names."""
    return sorted(__all__)
