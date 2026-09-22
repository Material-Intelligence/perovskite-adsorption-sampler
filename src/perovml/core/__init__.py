"""Core PerovML algorithms and placement strategies."""

from __future__ import annotations

from typing import Any

from perovml.core.placement import PbAdsorbateSlabConfig
from perovml.core.recipes import (
    dual_sampling_pipeline,
    perov_adslab_generator,
    perov_ml_pipeline,
    relax_job,
    run_perovml,
    select_best_by_priority,
)

__all__ = [
    "PbAdsorbateSlabConfig",
    "dual_sampling_pipeline",
    "perov_adslab_generator",
    "perov_ml_pipeline",
    "relax_job",
    "run_adsorption_task",
    "run_dft_stage",
    "run_perovml",
    "select_best_by_priority",
]

# perovml.recipes.adsorption imports perovml.core.recipes, so importing it here
# eagerly would make `import perovml.recipes.adsorption` fail with a circular
# import. Resolve those two names lazily instead (PEP 562), the same way
# perovml/__init__.py does.
_LAZY_IMPORTS: dict[str, str] = {
    "run_adsorption_task": "perovml.recipes.adsorption",
    "run_dft_stage": "perovml.recipes.adsorption",
}


def __getattr__(name: str) -> Any:
    """Import a public symbol on first access.

    Args:
        name: Attribute being looked up on ``perovml.core``.

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
