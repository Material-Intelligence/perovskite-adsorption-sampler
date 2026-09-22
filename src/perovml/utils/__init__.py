"""Structure-preparation and POSCAR utilities."""

from __future__ import annotations

from typing import Any

from perovml.utils.poscar_tools import selective_dynamics_by_z, uniform_selective_dynamics

__all__ = [
    "build_adsorbate",
    "build_oc_slab_from_atoms",
    "ensure_surface_tags_and_constraints",
    "selective_dynamics_by_z",
    "uniform_selective_dynamics",
]

# perovml.utils.structure imports FAIRChem; resolve its names lazily so that the
# POSCAR helpers stay importable without it.
_LAZY_IMPORTS: dict[str, str] = {
    "build_adsorbate": "perovml.utils.structure",
    "build_oc_slab_from_atoms": "perovml.utils.structure",
    "ensure_surface_tags_and_constraints": "perovml.utils.structure",
}


def __getattr__(name: str) -> Any:
    """Import a public symbol on first access.

    Args:
        name: Attribute being looked up on ``perovml.utils``.

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
