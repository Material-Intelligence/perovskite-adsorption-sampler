"""Adapt raw slab and molecule structures into FAIRChem ``Slab`` / ``Adsorbate`` objects.

A slab read from a POSCAR carries neither surface tags nor constraints, both of
which the FAIRChem ``Slab`` constructor requires. The helpers here fill that gap:

  - :func:`ensure_surface_tags_and_constraints` tags the top Pb-I layer as surface
    (``tag=1``) and fixes the subsurface atoms,
  - :func:`build_oc_slab_from_atoms` wraps the tagged atoms in a ``Slab``,
  - :func:`build_adsorbate` reads a molecule file, optionally guesses the binding
    atom, pre-orients it along ``+z`` and wraps it in an ``Adsorbate``.

The command-line front end for these helpers lives in :mod:`perovml.cli.place`
(``perovml place``).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from ase.atoms import Atoms
from ase.constraints import FixAtoms
from ase.data import atomic_numbers, covalent_radii
from ase.io import read

# Import directly from modules to avoid side-effects of package __init__ (which can pull core deps)
from fairchem.data.oc.core.adsorbate import Adsorbate
from fairchem.data.oc.core.slab import Slab

logger = logging.getLogger(__name__)

__all__ = ["build_adsorbate", "build_oc_slab_from_atoms", "ensure_surface_tags_and_constraints"]


def _tag_surface_pbi(atoms: Atoms, z_gap: float = 2.0, cluster_tol: float = 2.5) -> list[int] | None:
    """Tag the top Pb-I layer as surface (``tag=1``) from a z threshold.

    The top set is taken as the Pb and I atoms with ``z >= z_max - z_gap``. When
    the resulting Pb:I ratio is far from the expected 1:2, the selection is
    refined by clustering the Pb/I atoms along z and keeping the topmost clusters.

    Args:
        atoms: Slab structure.
        z_gap: Depth below the topmost Pb/I atom, in A, that still counts as surface.
        cluster_tol: Maximum z separation, in A, within one cluster.

    Returns:
        A tag list of length ``len(atoms)``, 1 for surface Pb/I atoms and 0
        elsewhere, or None when the structure contains neither Pb nor I.
    """
    Z_PB = 82
    Z_I = 53
    zs = atoms.get_atomic_numbers()
    pos = atoms.get_positions()
    n = len(atoms)
    idx_pb = [i for i, z in enumerate(zs) if z == Z_PB]
    idx_i = [i for i, z in enumerate(zs) if z == Z_I]
    if len(idx_pb) == 0 and len(idx_i) == 0:
        return None

    # candidate by z-gap
    zi = np.array([pos[i, 2] for i in (idx_pb + idx_i)])
    all_idx = np.array(idx_pb + idx_i)
    z_max = zi.max()
    sel = all_idx[zi >= (z_max - z_gap)]

    def ratio_good(pb_count, i_count):
        if i_count == 0 or pb_count == 0:
            return False
        r = pb_count / i_count
        return abs(r - 0.5) <= 0.5  # loose acceptance around 1:2

    def apply_and_return(sel_idx):
        tags = [0] * n
        for j in sel_idx:
            if zs[j] in (Z_PB, Z_I):
                tags[j] = 1
        return tags

    pb_count = sum(1 for j in sel if zs[j] == Z_PB)
    i_count = sum(1 for j in sel if zs[j] == Z_I)
    if ratio_good(pb_count, i_count):
        return apply_and_return(sel)

    # refine by z-clustering (descending), choose top clusters to approach 1:2
    pbi_idx = np.array(idx_pb + idx_i)
    zvals = np.array([pos[i, 2] for i in pbi_idx])
    order = np.argsort(-zvals)
    pbi_idx = pbi_idx[order]
    zvals = zvals[order]
    clusters = []
    cur = [pbi_idx[0]]
    for k in range(1, len(pbi_idx)):
        if abs(zvals[k] - zvals[k - 1]) <= cluster_tol:
            cur.append(pbi_idx[k])
        else:
            clusters.append(cur)
            cur = [pbi_idx[k]]
    if cur:
        clusters.append(cur)

    best = None
    target = 0.5
    for take in range(1, min(3, len(clusters)) + 1):
        sel2 = [j for cl in clusters[:take] for j in cl]
        pb2 = sum(1 for j in sel2 if zs[j] == Z_PB)
        i2 = sum(1 for j in sel2 if zs[j] == Z_I)
        if i2 == 0 or pb2 == 0:
            continue
        r = pb2 / i2
        score = abs(r - target)
        if (best is None) or (score < best[0]):
            best = (score, sel2)

    if best is not None:
        return apply_and_return(best[1])
    # fallback: use initial selection even if ratio off
    return apply_and_return(sel)


def ensure_surface_tags_and_constraints(
    atoms: Atoms,
    surface_fraction: float = 0.25,
    strict: bool = False,
) -> Atoms:
    """Add the surface tags and constraints that the FAIRChem ``Slab`` class requires.

    If every tag is 0, the top Pb-I layer is tagged as surface (``tag=1``); when the
    structure has no Pb or I the top ``surface_fraction`` of atoms by z is used
    instead. If the structure carries no constraints, all subsurface atoms are fixed.
    The periodic boundary flags are normalised to fully periodic, because several
    downstream components reject a partially periodic cell.

    Which of those routes was taken is recorded in ``atoms.info["surface_tagging"]``:
    ``"preserved"`` (the structure arrived tagged), ``"pb_i_layer"`` or
    ``"top_z_fraction"``. The last one is a geometric guess that knows nothing about the
    chemistry, so a slab it was applied to deserves a look before its numbers are trusted.

    Args:
        atoms: Slab structure. Modified in place.
        surface_fraction: Fraction of atoms (by z) treated as surface in the
            fallback path.
        strict: Refuse to guess. The structure must already carry surface tags.

    Returns:
        The same ``Atoms`` object, tagged and constrained.

    Raises:
        ValueError: If ``strict`` is set and the structure carries no tags.
    """
    tags = np.array(atoms.get_tags())
    method = "preserved"
    if (tags == 0).all():
        if strict:
            raise ValueError(
                "strict=True, but this structure carries no surface tags (every tag is 0). "
                "Tag the surface atoms with 1 before passing it in, or drop strict to let "
                "the Pb-I heuristic do it."
            )
        # Try Pb-I specific tagging first
        pbi_tags = _tag_surface_pbi(atoms)
        if pbi_tags is not None and any(t == 1 for t in pbi_tags):
            atoms.set_tags(pbi_tags)
            method = "pb_i_layer"
            # optional: print ratio info
            zs_num = atoms.get_atomic_numbers()
            pb_s = sum(1 for i, t in enumerate(pbi_tags) if t == 1 and zs_num[i] == 82)
            i_s = sum(1 for i, t in enumerate(pbi_tags) if t == 1 and zs_num[i] == 53)
            if i_s > 0:
                logger.info("Pb-I surface tagging: Pb:I = %d:%d (~%.2f)", pb_s, i_s, pb_s / max(i_s, 1))
        else:
            # Fallback: top fraction by z for all atoms
            z = atoms.get_positions()[:, 2]
            thresh = np.quantile(z, 1.0 - surface_fraction)
            new_tags = np.where(z >= thresh, 1, 0)
            atoms.set_tags(new_tags)
            method = "top_z_fraction"
            logger.warning(
                "no Pb-I termination found; tagging the top %.0f%% of atoms by z as the surface. "
                "This is a geometric guess -- check that it matches the slab you meant.",
                100 * surface_fraction,
            )

    info = atoms.info or {}
    info["surface_tagging"] = method
    atoms.info = info

    if len(atoms.constraints) == 0:
        new_tags = atoms.get_tags()
        fix_idx = [i for i, t in enumerate(new_tags) if t != 1]
        if len(fix_idx) > 0:
            atoms.set_constraint(FixAtoms(indices=fix_idx))

    # Ensure a consistent PBC flag. Downstream components such as UMA and FAIRChem
    # infer periodicity from the flags and require them to be either fully periodic
    # (True, True, True) or fully aperiodic (False, False, False); anything in
    # between raises "Attempted to guess PBC ... some dimensions but not others".
    pbc = atoms.get_pbc()
    if not pbc.any():
        # No PBC set: treat the slab as 3D periodic, the vacuum along z takes care of the rest.
        atoms.set_pbc((True, True, True))
    elif not pbc.all():
        # Partially periodic: promote to fully periodic.
        atoms.set_pbc((True, True, True))
    return atoms


def build_oc_slab_from_atoms(
    atoms: Atoms,
    min_ab: float,
    surface_fraction: float = 0.25,
    strict_tags: bool = False,
) -> Slab:
    """Wrap a raw slab structure in a FAIRChem ``Slab``.

    Args:
        atoms: Slab structure, with or without tags and constraints.
        min_ab: Minimum in-plane lattice vector length the ``Slab`` constructor
            requires; a warning is logged when the cell is smaller.
        surface_fraction: Forwarded to
            :func:`ensure_surface_tags_and_constraints`. Only used for an untagged
            structure in which the Pb-I tagging finds no surface layer.
        strict_tags: Require the structure to arrive already tagged, instead of letting
            the heuristics decide which atoms are the surface.

    Returns:
        The constructed ``Slab``. How its surface was identified is recorded in
        ``slab.atoms.info["surface_tagging"]``.
    """
    # The Slab constructor checks: min_ab on a,b, has_surface_tagged, constraints non-empty.
    atoms = ensure_surface_tags_and_constraints(atoms, surface_fraction=surface_fraction, strict=strict_tags)
    # (Optional) sanity on a,b lengths
    cell = atoms.cell
    a_len = np.linalg.norm(cell[0])
    b_len = np.linalg.norm(cell[1])
    if a_len < min_ab or b_len < min_ab:
        logger.warning(
            "a/b lengths are small (%.2f, %.2f) < %s; the Slab constructor may reject this cell",
            a_len,
            b_len,
            min_ab,
        )
    return Slab(bulk=None, slab_atoms=atoms, min_ab=min_ab)


def _guess_binding_index(mol: Atoms) -> tuple[int, dict]:
    """Guess which atom of a molecule binds to the surface.

    The rule is deterministic:

      1. Build a connectivity graph from interatomic distances, using the sum of
         the covalent radii plus a 0.2 A margin as the bond cutoff.
      2. Prefer hetero atoms, in the order O > N > S > P > F > Cl > Br > I.
      3. Among those, pick the one with the smallest degree, so a terminal atom
         wins over a bridging one. Ties are broken by the priority order above,
         then by atom index.

    Args:
        mol: Molecule to inspect.

    Returns:
        A tuple of the chosen atom index and a dict recording why it was chosen.
    """
    zs = mol.get_atomic_numbers()
    pos = mol.get_positions()
    n = len(zs)
    deg = [0] * n
    margin = 0.2
    # Build adjacency by simple distance threshold (no PBC for molecules)
    for i in range(n):
        for j in range(i + 1, n):
            th = covalent_radii[zs[i]] + covalent_radii[zs[j]] + margin
            if th <= 0:  # fallback if radius unknown
                continue
            if np.linalg.norm(pos[i] - pos[j]) < th:
                deg[i] += 1
                deg[j] += 1

    # Priority list of hetero atoms
    pri_list = ["O", "N", "S", "P", "F", "Cl", "Br", "I"]
    pri_map = {atomic_numbers[sym]: idx for idx, sym in enumerate(pri_list)}

    # Build candidate list with (priority, degree, index)
    candidates = []
    for i, z in enumerate(zs):
        pri = pri_map.get(z, 999)  # non-hetero get lowest priority
        candidates.append((pri, deg[i], i))

    # Prefer hetero (pri small), then degree small, then index small
    candidates.sort()
    chosen_pri, chosen_deg, chosen_idx = candidates[0]

    meta = {
        "chosen_index": int(chosen_idx),
        "chosen_element": int(zs[chosen_idx]),
        "degree": int(chosen_deg),
        "priority": int(chosen_pri),
        "priority_order": pri_list,
    }
    return chosen_idx, meta


def _get_bonded_neighbors(mol: Atoms, binding_idx: int, margin: float = 0.2) -> list[int]:
    """List the atoms bonded to the binding atom.

    Two atoms count as bonded when their distance is below the sum of their
    covalent radii plus ``margin``.

    Args:
        mol: Molecule to inspect.
        binding_idx: Index of the binding atom.
        margin: Extra allowance on the covalent-radii sum, in A.

    Returns:
        The indices of the bonded neighbours.
    """
    zs = mol.get_atomic_numbers()
    pos = mol.get_positions()
    z0 = zs[binding_idx]
    p0 = pos[binding_idx]
    neigh = []
    for j in range(len(mol)):
        if j == binding_idx:
            continue
        th = covalent_radii[z0] + covalent_radii[zs[j]] + margin
        if th <= 0:
            continue
        if np.linalg.norm(pos[j] - p0) < th:
            neigh.append(j)
    return neigh


def _normalize(v: np.ndarray) -> np.ndarray:
    """Return the unit vector along ``v``, or ``v`` itself when it is degenerate.

    Args:
        v: Vector to normalise.

    Returns:
        The normalised vector.
    """
    n = np.linalg.norm(v)
    return v if n < 1e-12 else v / n


def _rotation_matrix_from_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Build the Rodrigues rotation matrix that maps ``a`` onto ``b``.

    Args:
        a: Source vector; need not be normalised.
        b: Target vector; need not be normalised.

    Returns:
        A 3x3 rotation matrix.
    """
    a_n = _normalize(a)
    b_n = _normalize(b)
    v = np.cross(a_n, b_n)
    c = float(np.clip(np.dot(a_n, b_n), -1.0, 1.0))
    s = np.linalg.norm(v)
    if s < 1e-12:
        if c > 0:
            return np.eye(3)
        # 180-degree: pick arbitrary orthogonal axis
        ref = np.array([1.0, 0.0, 0.0])
        if abs(np.dot(a_n, ref)) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        axis = _normalize(np.cross(a_n, ref))
        K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
        return np.eye(3) + K @ K * -1.0
    k = v / s
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + K * s + K @ K * (1 - c)


def _pre_orient_binding(mol: Atoms, binding_idx: int) -> Atoms:
    """Rotate a molecule so that the bonds around its binding atom point along +z.

    The rotation is applied about the binding atom, so the binding atom itself
    does not move and all internal geometry is preserved. The direction that is
    mapped onto +z is:

      - one neighbour: the binding-to-neighbour vector;
      - two or three neighbours: the sum of the unit bond vectors;
      - if that sum vanishes: any vector perpendicular to the line for two
        neighbours, or the plane normal from an SVD for three or more.

    Args:
        mol: Molecule to orient. It is not modified.
        binding_idx: Index of the binding atom.

    Returns:
        A rotated copy, or ``mol`` itself when the binding atom has no neighbours
        or no well-defined direction could be found.
    """
    neigh = _get_bonded_neighbors(mol, binding_idx)
    if not neigh:
        return mol
    p0 = mol.get_positions()[binding_idx]
    vecs = [mol.get_positions()[j] - p0 for j in neigh]
    z_dir = None
    if len(neigh) == 1:
        z_dir = _normalize(vecs[0])
    else:
        v_sum = np.sum([_normalize(v) for v in vecs], axis=0)
        if np.linalg.norm(v_sum) > 1e-6:
            z_dir = _normalize(v_sum)
        else:
            if len(neigh) == 2:
                v = _normalize(vecs[0])
                ref = np.array([1.0, 0.0, 0.0])
                if abs(np.dot(v, ref)) > 0.9:
                    ref = np.array([0.0, 1.0, 0.0])
                z_dir = _normalize(np.cross(v, ref))
            else:
                M = np.stack(vecs, axis=0)
                M = M - M.mean(axis=0)
                try:
                    _, _, vh = np.linalg.svd(M, full_matrices=False)
                    normal = vh[-1]
                except Exception:
                    normal = np.cross(vecs[0], vecs[1])
                z_dir = _normalize(normal)
    if z_dir is None or np.linalg.norm(z_dir) < 1e-9:
        return mol
    R = _rotation_matrix_from_a_to_b(z_dir, np.array([0.0, 0.0, 1.0]))
    pos = mol.get_positions()
    pos_shift = pos - p0
    pos_rot = (R @ pos_shift.T).T + p0
    mol_oriented = mol.copy()
    mol_oriented.set_positions(pos_rot)
    return mol_oriented


def build_adsorbate(
    molecule_file: str | Path | None = None,
    adsorbate: str | None = None,
    binding_index: int | None = None,
    run_dir: str | Path | None = None,
) -> Adsorbate:
    """Build a FAIRChem ``Adsorbate`` from a molecule file or a database key.

    When ``molecule_file`` is given the molecule is read with ASE, its periodic
    boundary flags are normalised to fully aperiodic, the binding atom is guessed
    if it was not supplied, and the molecule is pre-oriented so that the bonds
    around the binding atom point along ``+z``.

    Args:
        molecule_file: Path to an ASE-readable molecule file. Takes precedence
            over ``adsorbate``.
        adsorbate: Adsorbate key from the FAIRChem database (e.g. ``"*O"``).
            Used only when ``molecule_file`` is None.
        binding_index: Index of the binding atom. Guessed heuristically when None.
        run_dir: Optional directory in which to append a ``binding_guess.txt``
            audit record whenever the binding index is guessed.

    Returns:
        The constructed ``Adsorbate``.

    Raises:
        ValueError: If neither ``molecule_file`` nor ``adsorbate`` is given.
    """
    if molecule_file is None:
        if adsorbate is None:
            raise ValueError("build_adsorbate requires either `molecule_file` or `adsorbate`.")
        return Adsorbate(adsorbate_smiles_from_db=adsorbate)

    mol = read(str(molecule_file))
    # Normalise the molecule's PBC flags: a partially periodic cell makes UMA and ASE
    # raise when they try to guess the periodicity, and a gas-phase molecule is
    # better described as fully aperiodic anyway.
    pbc = mol.get_pbc()
    if pbc.any() and not pbc.all():
        mol.set_pbc((False, False, False))

    if binding_index is None:
        guess_idx, meta = _guess_binding_index(mol)
        binding_index = int(guess_idx)
        logger.info("guessed binding index = %d (Z=%d)", binding_index, mol.get_atomic_numbers()[binding_index])
        # Optionally save guess meta for audit
        if run_dir is not None:
            try:
                with open(Path(run_dir) / "binding_guess.txt", "a") as f:
                    f.write(f"molecule={molecule_file}, meta={meta}\n")
            except OSError:
                pass

    # Pre-orient molecule irrespective of placement mode
    mol = _pre_orient_binding(mol, binding_index)
    return Adsorbate(adsorbate_atoms=mol, adsorbate_binding_indices=[binding_index])
