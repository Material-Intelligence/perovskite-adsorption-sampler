"""Adsorbate placement on a Pb-I terminated perovskite surface.

The module holds one class, :class:`PbAdsorbateSlabConfig`, which subclasses FAIRChem's
``AdsorbateSlabConfig`` and replaces its site selection and orientation sampling. Both of its
modes commit to a single adsorption site -- the surface Pb atom nearest the cell centre under a
minimum-image metric in fractional *xy* -- and differ only in how the molecule is oriented and
anchored above it.

The algorithm, including the rotation sampling and the lifting rule, is written out in
``docs/pb_sampling_algorithm.md``.
"""

from __future__ import annotations

import math
from typing import List

import numpy as np
from ase.data import covalent_radii
from ase.geometry import get_distances
from fairchem.data.oc.core.adsorbate_slab_config import AdsorbateSlabConfig

MODES = ("Pb_uniform_sample", "Pb_heuristic_sample")
"""The sampling modes :class:`PbAdsorbateSlabConfig` accepts."""

_GOLDEN_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
"""Azimuthal increment of a Fibonacci spiral, in radians."""

OVERLAP_TOLERANCE = 0.5
"""How far inside the sum of two covalent radii a contact may sit, in Angstrom."""


class PbAdsorbateSlabConfig(AdsorbateSlabConfig):
    """Sample adsorbate orientations above the central surface Pb atom.

    Two modes are available, both placing at one site:

    - ``"Pb_uniform_sample"``: the molecule is anchored at its centre of mass and rotated by
      uniform SO(3) rotations drawn from Marsaglia quaternions. Use it when the binding
      geometry is unknown.
    - ``"Pb_heuristic_sample"``: the molecule is anchored at its binding atom and oriented on a
      deterministic spiral covering the whole spherical cap of half-angle ``cone_angle_deg``
      about the surface normal, from the upright geometry on the axis out to the rim. Use it
      when the binding geometry is known.

    Both lift the molecule along the normal by a computed interstitial gap and tag the merged
    adslab (bulk ``0``, surface ``1``, adsorbate ``2``) the way downstream FAIRChem code expects.

    Attributes:
        sites: The single chosen site, as a one-element list, for compatibility with the base
            class.
        atoms_list: The generated adslab structures, one per orientation.
        metadata_list: Per-structure metadata, in the same order as ``atoms_list``.
        seed: The integer seed the random orientations were drawn from, or None when the
            caller supplied its own ``numpy`` generator. Record it to reproduce a run.
    """

    def __init__(
        self,
        slab,
        adsorbate,
        num_orientations: int = 100,
        cone_angle_deg: float = 20.0,
        interstitial_gap: float = 0.1,
        mode: str = "Pb_uniform_sample",
        rng: np.random.Generator | int | None = None,
    ):
        """Pick the site and generate every orientation immediately.

        Sampling happens in the constructor, as in the base class: once it returns,
        ``atoms_list`` holds the finished structures.

        Args:
            slab: FAIRChem ``Slab`` whose surface atoms are already tagged. Build one with
                :func:`~perovml.utils.structure.build_oc_slab_from_atoms`.
            adsorbate: FAIRChem ``Adsorbate``, carrying the molecule and its binding index.
            num_orientations: Orientations to generate. Used by both modes.
            cone_angle_deg: Half-angle of the cone in degrees. ``Pb_heuristic_sample`` only.
            interstitial_gap: Extra clearance in Angstrom kept when lifting the molecule along
                the surface normal.
            mode: Either ``"Pb_uniform_sample"`` or ``"Pb_heuristic_sample"``.
            rng: Seed or ``numpy`` generator for ``Pb_uniform_sample``. An integer makes the
                run reproducible; None draws a fresh seed and records it in :attr:`seed`.
                ``Pb_heuristic_sample`` is deterministic and ignores it.

        Raises:
            ValueError: If ``mode`` is not one of the two supported names, or if a generated
                structure buries the molecule in the slab.
        """
        if mode not in MODES:
            raise ValueError(f"mode must be one of {list(MODES)}, got {mode!r}")
        self._pmode = mode
        self._N = int(num_orientations)
        self._theta = float(cone_angle_deg)

        # A concrete seed is always recorded, so that a run can be repeated from its metadata
        # even when the caller did not supply one.
        if isinstance(rng, np.random.Generator):
            self.seed: int | None = None
            self._rng = rng
        else:
            self.seed = int(rng) if rng is not None else int(np.random.SeedSequence().entropy % (2**63))
            self._rng = np.random.default_rng(self.seed)

        # Set base fields required by parent
        self.slab = slab
        self.adsorbate = adsorbate
        self.num_sites = 1
        self.num_augmentations_per_site = self._N
        self.interstitial_gap = interstitial_gap
        self.mode = mode

        # Build single site (central Pb)
        self.sites = [self._get_central_pb_site()]
        self.atoms_list, self.metadata_list = self._place_with_custom_orientations()

    # -------- utility rotation helpers --------
    @staticmethod
    def _normalize(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v if n < 1e-12 else v / n

    @staticmethod
    def _rot_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
        k = PbAdsorbateSlabConfig._normalize(axis)
        K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
        c = math.cos(angle)
        s = math.sin(angle)
        return np.eye(3) + K * s + K @ K * (1 - c)

    @staticmethod
    def _rot_a_to_b(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        a = PbAdsorbateSlabConfig._normalize(a)
        b = PbAdsorbateSlabConfig._normalize(b)
        v = np.cross(a, b)
        c = float(np.clip(np.dot(a, b), -1.0, 1.0))
        s = np.linalg.norm(v)
        if s < 1e-12:
            if c > 0:
                return np.eye(3)
            ref = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(a, ref)) > 0.9:
                ref = np.array([0.0, 1.0, 0.0])
            return PbAdsorbateSlabConfig._rot_axis_angle(np.cross(a, ref), math.pi)
        return PbAdsorbateSlabConfig._rot_axis_angle(v, math.atan2(s, c))

    # -------- site finder --------
    def _get_central_pb_site(self) -> np.ndarray:
        Z_PB = 82
        atoms = self.slab.atoms
        scaled = atoms.get_scaled_positions()
        zs = atoms.get_atomic_numbers()
        tags = atoms.get_tags()
        # Choose surface Pb (tag==1) closest to (0.5, 0.5) in xy fractional, torus metric
        pb_idx: List[int] = [i for i, (z, t) in enumerate(zip(zs, tags)) if z == Z_PB and t == 1]
        if not pb_idx:
            raise ValueError("No surface Pb atom found (tag==1 & Z==82). Ensure tagging is correct.")

        def torus_dist_xy(frac_xy: np.ndarray) -> float:
            # distance on torus to (0.5, 0.5)
            d = np.abs(frac_xy - np.array([0.5, 0.5]))
            d = np.minimum(d, 1.0 - d)
            return float(np.linalg.norm(d))

        best_i = min(pb_idx, key=lambda i: torus_dist_xy(scaled[i][:2]))
        return atoms.get_positions()[best_i]

    # -------- sampling --------
    def _sample_uniform_so3(self, N: int) -> List[np.ndarray]:
        # Marsaglia method using quaternions, drawn from this instance's generator so that
        # the candidate set is reproducible from the recorded seed.
        R = []
        for _ in range(N):
            u1, u2, u3 = self._rng.random(3)
            q1 = math.sqrt(1 - u1) * math.sin(2 * math.pi * u2)
            q2 = math.sqrt(1 - u1) * math.cos(2 * math.pi * u2)
            q3 = math.sqrt(u1) * math.sin(2 * math.pi * u3)
            q4 = math.sqrt(u1) * math.cos(2 * math.pi * u3)
            # quaternion to rotation
            x, y, z, w = q1, q2, q3, q4
            Rm = np.array(
                [
                    [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                    [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                    [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
                ]
            )
            R.append(Rm)
        return R

    def _sample_cone_grid(self, N: int, theta_deg: float) -> List[np.ndarray]:
        """Cover the spherical cap of half-angle ``theta_deg`` with exactly ``N`` orientations.

        Each orientation tilts ``+z`` onto a direction inside the cap and then twists the
        molecule about that tilted axis. The polar angle runs from 0 (upright on the surface
        normal, the geometry a chemist expects first) out to ``theta_deg`` on the rim, spaced
        evenly in ``cos(beta)`` so the samples are spread by cap area rather than by angle.
        The azimuth follows a golden-angle spiral, so no two samples share a meridian.

        An earlier version built a full ``beta x psi x phi`` grid and truncated it to ``N``.
        Because ``beta`` was the outer loop, the truncation kept only the first band and every
        returned orientation sat on the rim; the upright geometry was never generated.

        Args:
            N: Number of orientations to return.
            theta_deg: Half-angle of the cone, in degrees.

        Returns:
            ``N`` rotation matrices.
        """
        theta = math.radians(theta_deg)
        cos_theta = math.cos(theta)
        z_axis = np.array([0.0, 0.0, 1.0])

        R_list: List[np.ndarray] = []
        for i in range(N):
            # 0 on the cone axis, 1 on its rim.
            frac = 0.0 if N == 1 else i / (N - 1)
            cos_beta = float(np.clip(1.0 - frac * (1.0 - cos_theta), -1.0, 1.0))
            beta = math.acos(cos_beta)
            psi = i * _GOLDEN_ANGLE
            u = np.array(
                [
                    math.sin(beta) * math.cos(psi),
                    math.sin(beta) * math.sin(psi),
                    cos_beta,
                ]
            )
            R_align = self._rot_a_to_b(z_axis, u)
            R_twist = self._rot_axis_angle(u, 2 * math.pi * i / N)
            R_list.append(R_twist @ R_align)
        return R_list

    # -------- placement --------
    def _place_with_custom_orientations(self):
        atoms_list = []
        meta_list = []
        site = self.sites[0]
        # Surface normal. np.cross(a, b) points into the slab for a left-handed cell, which
        # some POSCARs do carry, so orient it away from the slab's centre of mass before it is
        # used to lift the molecule.
        cell = self.slab.atoms.cell
        normal = np.cross(cell[0], cell[1])
        unit_normal = self._normalize(normal)
        outward = site - self.slab.atoms.get_center_of_mass()
        reference = outward if np.linalg.norm(outward) > 1e-6 else np.array([0.0, 0.0, 1.0])
        if float(np.dot(unit_normal, reference)) < 0.0:
            unit_normal = -unit_normal

        # choose rotations
        if self._pmode == "Pb_uniform_sample":
            rotations = self._sample_uniform_so3(self._N)
            anchor_mode = "com"
        else:
            rotations = self._sample_cone_grid(self._N, self._theta)
            anchor_mode = "binding"

        # binding index if needed
        if anchor_mode == "binding":
            if not self.adsorbate.binding_indices:
                raise ValueError("Binding indices not set on adsorbate for Pb_heuristic_sample")
            binding_idx = int(self.adsorbate.binding_indices[0])

        for R in rotations:
            # orient copy
            ads_c = self.adsorbate.atoms.copy()
            # anchor point
            if anchor_mode == "com":
                anchor = ads_c.get_center_of_mass()
            else:
                anchor = ads_c.get_positions()[binding_idx]
            pos = ads_c.get_positions()
            pos_shift = pos - anchor
            pos_rot = (R @ pos_shift.T).T + anchor
            ads_c.set_positions(pos_rot)

            # translate to site
            placement_center = anchor
            ads_c.translate(site - placement_center)

            # lift along normal to avoid overlaps
            lift = self._get_scaled_normal(ads_c, self.slab.atoms.copy(), site, unit_normal, self.interstitial_gap)
            ads_c.translate(lift * unit_normal)

            # merge
            slab_c = self.slab.atoms.copy()
            adslab = slab_c + ads_c
            tags = list(slab_c.get_tags()) + [2] * len(ads_c)
            adslab.set_tags(tags)
            adslab.cell = slab_c.cell
            adslab.pbc = [True, True, False]

            self._assert_no_overlap(adslab, len(slab_c))

            atoms_list.append(adslab)
            meta_list.append({"site": site, "R": R})

        return atoms_list, meta_list

    @staticmethod
    def _assert_no_overlap(adslab, n_slab: int) -> None:
        """Fail when the adsorbate ended up inside the slab rather than above it.

        The lift along the surface normal is meant to leave at least the sum of the two
        covalent radii between any adsorbate atom and any slab atom. A contact far inside that
        sum means the placement went wrong -- a normal pointing the wrong way, a degenerate
        cell -- and such a structure must not reach a relaxation as if it were a candidate.

        Args:
            adslab: The merged structure, slab atoms first.
            n_slab: Number of slab atoms, i.e. where the adsorbate block starts.

        Raises:
            ValueError: If a contact sits more than :data:`OVERLAP_TOLERANCE` inside the sum
                of the two covalent radii.
        """
        positions = adslab.get_positions()
        numbers = adslab.get_atomic_numbers()
        slab_pos, ads_pos = positions[:n_slab], positions[n_slab:]
        if len(slab_pos) == 0 or len(ads_pos) == 0:
            return

        _, distances = get_distances(ads_pos, slab_pos, cell=adslab.cell, pbc=adslab.pbc)
        radii_sum = covalent_radii[numbers[n_slab:]][:, None] + covalent_radii[numbers[:n_slab]][None, :]
        clearance = distances - radii_sum
        worst = int(np.argmin(clearance))
        if clearance.flat[worst] < -OVERLAP_TOLERANCE:
            ads_i, slab_i = np.unravel_index(worst, clearance.shape)
            raise ValueError(
                f"adsorbate atom {int(ads_i)} sits {abs(float(clearance.flat[worst])):.2f} A inside the "
                f"covalent contact distance of slab atom {int(slab_i)} "
                f"({float(distances[ads_i, slab_i]):.2f} A apart); the molecule was buried rather than "
                "placed on the surface"
            )
