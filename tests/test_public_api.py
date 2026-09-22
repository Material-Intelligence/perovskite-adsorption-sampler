"""Tests for the package's advertised public surface.

``perovml`` resolves most of :data:`perovml.__all__` lazily (PEP 562), which means a name can
stay listed long after the module behind it stopped importing. These tests resolve every name,
so that ``__all__`` cannot drift away from what the package can actually hand out.
"""

from __future__ import annotations

import importlib

import pytest

import perovml

#: Subpackages that must import on their own, without going through a lazy attribute.
SUBPACKAGES = [
    "perovml.calculators",
    "perovml.cli",
    "perovml.core",
    "perovml.dft",
    "perovml.parallel",
    "perovml.recipes",
    "perovml.utils",
]


@pytest.mark.parametrize("name", sorted(perovml.__all__))
def test_every_advertised_name_resolves(name):
    """Each entry in ``__all__`` is reachable as an attribute of the package."""
    assert getattr(perovml, name) is not None


@pytest.mark.parametrize("module_name", SUBPACKAGES)
def test_every_subpackage_imports(module_name):
    """Each shipped subpackage imports cleanly against the pinned dependencies."""
    assert importlib.import_module(module_name) is not None


def test_unknown_name_raises_attribute_error():
    """A name outside ``__all__`` is an ``AttributeError``, not a silent None."""
    with pytest.raises(AttributeError):
        perovml.NotAPublicName  # noqa: B018


def test_dir_lists_the_public_api():
    """``dir(perovml)`` reports exactly the advertised names."""
    assert sorted(dir(perovml)) == sorted(perovml.__all__)
