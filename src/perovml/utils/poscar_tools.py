"""Selective-dynamics helpers for VASP POSCAR files.

Slab calculations usually freeze the bottom of the slab and relax the top. VASP
expresses that through a ``Selective dynamics`` block: three ``T``/``F`` flags per
site. The helpers here build that block from a :class:`~pymatgen.core.Structure`
and hand back a ready-to-write :class:`~pymatgen.io.vasp.inputs.Poscar`.

Examples:
    >>> from pymatgen.io.vasp.inputs import Poscar  # doctest: +SKIP
    >>> from perovml.utils.poscar_tools import selective_dynamics_by_z  # doctest: +SKIP
    >>> poscar = Poscar.from_file("slab.vasp")  # doctest: +SKIP
    >>> out = selective_dynamics_by_z(poscar.structure, z_cut=0.26, comment=poscar.comment)  # doctest: +SKIP
    >>> out.write_file("slab_sd.vasp")  # doctest: +SKIP

The command-line front end is ``perovml sd``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from monty.dev import requires

if TYPE_CHECKING:  # pragma: no cover - typing only
    from pymatgen.core import Structure
    from pymatgen.io.vasp.inputs import Poscar

try:
    from pymatgen.io.vasp.inputs import Poscar  # noqa: F811

    pmg_installed = True
except ImportError:  # pragma: no cover - runtime import guard
    pmg_installed = False

__all__ = ["FIXED", "FREE", "XY_ONLY", "selective_dynamics_by_z", "uniform_selective_dynamics"]

#: All three Cartesian directions relaxed.
FREE: tuple[bool, bool, bool] = (True, True, True)
#: All three Cartesian directions frozen.
FIXED: tuple[bool, bool, bool] = (False, False, False)
#: In-plane motion only, z frozen.
XY_ONLY: tuple[bool, bool, bool] = (True, True, False)

_PMG_MESSAGE = "Requires `pymatgen` to be installed (pip install pymatgen)"


@requires(pmg_installed, message=_PMG_MESSAGE)
def uniform_selective_dynamics(
    structure: Structure,
    flags: tuple[bool, bool, bool] = XY_ONLY,
    comment: str | None = None,
) -> Poscar:
    """Give every site the same selective-dynamics flags.

    Args:
        structure: Structure to tag.
        flags: ``(x, y, z)`` flags applied to every site, where True means the
            direction is relaxed. Defaults to :data:`XY_ONLY` (``T T F``).
        comment: POSCAR title line. Defaults to pymatgen's own formula-based title.

    Returns:
        A ``Poscar`` whose ``selective_dynamics`` is set for every site.
    """
    selective_dynamics = [list(flags) for _ in structure]
    return Poscar(structure, comment=comment, selective_dynamics=selective_dynamics)


@requires(pmg_installed, message=_PMG_MESSAGE)
def selective_dynamics_by_z(
    structure: Structure,
    z_cut: float = 0.26,
    free_flags: tuple[bool, bool, bool] = FREE,
    fixed_flags: tuple[bool, bool, bool] = FIXED,
    comment: str | None = None,
) -> Poscar:
    """Set selective-dynamics flags from a fractional-z cutoff.

    Sites with fractional ``z > z_cut`` get ``free_flags``; every other site gets
    ``fixed_flags``. This is the usual way to relax the top of a slab while
    holding the bulk-like bottom fixed.

    Args:
        structure: Structure to tag.
        z_cut: Fractional-z threshold, in ``[0, 1]``.
        free_flags: Flags for sites above the cutoff. Defaults to :data:`FREE`.
        fixed_flags: Flags for sites at or below the cutoff. Defaults to :data:`FIXED`.
        comment: POSCAR title line. Defaults to pymatgen's own formula-based title.

    Returns:
        A ``Poscar`` whose ``selective_dynamics`` is set for every site.

    Raises:
        ValueError: If ``z_cut`` is outside ``[0, 1]``.
    """
    if not 0.0 <= z_cut <= 1.0:
        raise ValueError(f"z_cut must be a fractional coordinate in [0, 1], got {z_cut}")

    selective_dynamics = [list(free_flags) if site.frac_coords[2] > z_cut else list(fixed_flags) for site in structure]
    return Poscar(structure, comment=comment, selective_dynamics=selective_dynamics)
