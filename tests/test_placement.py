"""Tests for the two Pb-site adsorbate placement samplers.

The samplers are exercised through their public entry point,
:class:`perovml.core.placement.PbAdsorbateSlabConfig`, on the slab and probe
molecule shipped in ``data/``. Nothing here needs a machine-learning model, a
network connection or a GPU.
"""

from __future__ import annotations

import numpy as np
import pytest
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor

from perovml.core.placement import PbAdsorbateSlabConfig
from perovml.utils.structure import build_oc_slab_from_atoms

#: Number of orientations used for the statistical checks on the uniform sampler.
N_UNIFORM = 500

#: Seed handed to the uniform sampler, which makes its candidate set reproducible.
SEED = 20260922

Z_AXIS = np.array([0.0, 0.0, 1.0])


@pytest.fixture(scope="module")
def uniform_config(slab, adsorbate):
    """A ``Pb_uniform_sample`` run with a fixed seed, built once per module."""
    return PbAdsorbateSlabConfig(
        slab=slab,
        adsorbate=adsorbate,
        num_orientations=N_UNIFORM,
        mode="Pb_uniform_sample",
        rng=SEED,
    )


@pytest.fixture(scope="module")
def uniform_rotations(uniform_config) -> np.ndarray:
    """The rotation matrices of the uniform run, shape ``(N_UNIFORM, 3, 3)``."""
    return np.array([meta["R"] for meta in uniform_config.metadata_list])


def _orientations(rotations: np.ndarray) -> np.ndarray:
    """Map rotation matrices to the direction each sends ``+z`` to.

    Args:
        rotations: Rotation matrices, shape ``(n, 3, 3)``.

    Returns:
        Unit vectors, shape ``(n, 3)``.
    """
    return rotations @ Z_AXIS


def _polar_angles_deg(rotations: np.ndarray) -> np.ndarray:
    """Angle between each rotated ``+z`` axis and ``+z`` itself, in degrees.

    Args:
        rotations: Rotation matrices, shape ``(n, 3, 3)``.

    Returns:
        Angles in degrees, shape ``(n,)``.
    """
    cosines = np.clip(_orientations(rotations) @ Z_AXIS, -1.0, 1.0)
    return np.degrees(np.arccos(cosines))


class TestUniformSampler:
    """``Pb_uniform_sample`` draws proper rotations covering all of SO(3)."""

    def test_returns_the_requested_number_of_configurations(self, uniform_config):
        assert len(uniform_config.atoms_list) == N_UNIFORM
        assert len(uniform_config.metadata_list) == N_UNIFORM

    def test_rotations_are_orthogonal(self, uniform_rotations):
        products = np.transpose(uniform_rotations, (0, 2, 1)) @ uniform_rotations
        identity = np.broadcast_to(np.eye(3), products.shape)
        assert np.allclose(products, identity, atol=1e-10)

    def test_rotations_have_unit_positive_determinant(self, uniform_rotations):
        determinants = np.linalg.det(uniform_rotations)
        assert np.allclose(determinants, 1.0, atol=1e-10), "rotations must be proper, not reflections"

    def test_orientations_have_no_preferred_direction(self, uniform_rotations):
        directions = _orientations(uniform_rotations)
        assert np.allclose(np.linalg.norm(directions, axis=1), 1.0, atol=1e-10)
        # For N isotropic directions the mean vector has length ~1/sqrt(N);
        # 0.15 is well above that for N = 500 and well below any real bias.
        assert np.linalg.norm(directions.mean(axis=0)) < 0.15

    def test_orientations_reach_every_octant(self, uniform_rotations):
        directions = _orientations(uniform_rotations)
        octants = {tuple(np.sign(direction).astype(int)) for direction in directions}
        assert len(octants) == 8, f"only {len(octants)} octants sampled"

    def test_polar_angle_is_uniform_in_cosine(self, uniform_rotations):
        # For an isotropic distribution cos(theta) is uniform on [-1, 1], so each
        # of ten equal-width bins should hold about a tenth of the samples.
        cosines = np.cos(np.radians(_polar_angles_deg(uniform_rotations)))
        counts, _ = np.histogram(cosines, bins=10, range=(-1.0, 1.0))
        expected = N_UNIFORM / 10
        assert counts.min() > 0.4 * expected, f"bin counts {counts.tolist()} are far from uniform"
        assert counts.max() < 1.6 * expected, f"bin counts {counts.tolist()} are far from uniform"

    def test_orientations_are_not_confined_near_the_surface_normal(self, uniform_rotations):
        angles = _polar_angles_deg(uniform_rotations)
        assert angles.max() > 170.0
        assert angles.min() < 10.0


class TestHeuristicSampler:
    """``Pb_heuristic_sample`` covers the cone, from its axis out to its rim."""

    @pytest.mark.parametrize("cone_angle_deg", [10.0, 25.0, 45.0])
    def test_orientations_stay_inside_the_cone(self, slab, adsorbate, cone_angle_deg):
        config = PbAdsorbateSlabConfig(
            slab=slab,
            adsorbate=adsorbate,
            num_orientations=20,
            cone_angle_deg=cone_angle_deg,
            mode="Pb_heuristic_sample",
        )
        angles = _polar_angles_deg(np.array([meta["R"] for meta in config.metadata_list]))
        assert len(angles) == 20
        assert angles.max() <= cone_angle_deg + 1e-6
        # The rim is sampled, so the widest orientation sits on it.
        assert angles.max() == pytest.approx(cone_angle_deg, abs=1e-6)

    @pytest.mark.parametrize(("num_orientations", "cone_angle_deg"), [(8, 30.0), (12, 25.0), (20, 25.0), (50, 20.0)])
    def test_the_interior_of_the_cone_is_sampled_too(self, slab, adsorbate, num_orientations, cone_angle_deg):
        """The upright geometry must be generated, not only the rim of the cone.

        A sampler that returns the rim alone passes every bound check above while never
        producing the orientation a chemist would try first, so the coverage is asserted
        explicitly.
        """
        config = PbAdsorbateSlabConfig(
            slab=slab,
            adsorbate=adsorbate,
            num_orientations=num_orientations,
            cone_angle_deg=cone_angle_deg,
            mode="Pb_heuristic_sample",
        )
        angles = _polar_angles_deg(np.array([meta["R"] for meta in config.metadata_list]))
        assert angles.min() < 0.2 * cone_angle_deg, "no orientation near the surface normal"
        assert len(np.unique(np.round(angles, 3))) >= 3, f"only {len(np.unique(np.round(angles, 3)))} polar angles"
        assert angles.mean() < 0.8 * cone_angle_deg, "the orientations are bunched against the rim"

    def test_the_same_settings_give_the_same_orientations(self, slab, adsorbate):
        """The cone sampler is deterministic, so a repeat run reproduces it exactly."""
        rotations = [
            np.array(
                [
                    meta["R"]
                    for meta in PbAdsorbateSlabConfig(
                        slab=slab,
                        adsorbate=adsorbate,
                        num_orientations=12,
                        cone_angle_deg=25.0,
                        mode="Pb_heuristic_sample",
                    ).metadata_list
                ]
            )
            for _ in range(2)
        ]
        assert np.allclose(rotations[0], rotations[1])

    def test_rotations_are_proper(self, slab, adsorbate):
        config = PbAdsorbateSlabConfig(
            slab=slab,
            adsorbate=adsorbate,
            num_orientations=20,
            cone_angle_deg=25.0,
            mode="Pb_heuristic_sample",
        )
        rotations = np.array([meta["R"] for meta in config.metadata_list])
        products = np.transpose(rotations, (0, 2, 1)) @ rotations
        assert np.allclose(products, np.broadcast_to(np.eye(3), products.shape), atol=1e-10)
        assert np.allclose(np.linalg.det(rotations), 1.0, atol=1e-10)


class TestSurfaceSiteSelection:
    """The single adsorption site is the surface Pb closest to the cell centre."""

    @staticmethod
    def _synthetic_slab():
        """Build a slab whose expected adsorption site is known by construction.

        Four Pb atoms sit in the surface layer and one, deliberately the one
        closest to the centre in ``xy``, sits below it with ``tag=0``.

        Returns:
            A ``(slab, expected_site)`` tuple, the site in Cartesian coordinates.
        """
        lattice = Lattice.from_parameters(12.0, 12.0, 30.0, 90.0, 90.0, 90.0)
        species = ["Pb", "Pb", "Pb", "Pb", "I", "I"]
        frac_coords = [
            [0.46, 0.53, 0.80],  # surface Pb nearest the centre -> expected site
            [0.08, 0.12, 0.80],  # surface Pb in a corner
            [0.91, 0.87, 0.80],  # surface Pb in the opposite corner
            [0.50, 0.50, 0.40],  # subsurface Pb exactly at the centre -> must be ignored
            [0.25, 0.75, 0.80],
            [0.75, 0.25, 0.80],
        ]
        tags = [1, 1, 1, 0, 1, 1]
        structure = Structure(lattice, species, frac_coords, site_properties={"tags": tags})
        atoms = AseAtomsAdaptor.get_atoms(structure)
        expected_site = atoms.get_positions()[0].copy()
        return build_oc_slab_from_atoms(atoms, min_ab=10.0), expected_site

    def test_site_is_the_surface_pb_nearest_the_cell_centre(self, adsorbate):
        slab, expected_site = self._synthetic_slab()
        config = PbAdsorbateSlabConfig(
            slab=slab,
            adsorbate=adsorbate,
            num_orientations=2,
            mode="Pb_uniform_sample",
        )
        assert len(config.sites) == 1
        assert np.allclose(config.sites[0], expected_site)

    def test_slab_without_surface_pb_is_rejected(self, adsorbate):
        lattice = Lattice.from_parameters(12.0, 12.0, 30.0, 90.0, 90.0, 90.0)
        structure = Structure(
            lattice,
            ["I", "I", "I"],
            [[0.50, 0.50, 0.80], [0.25, 0.25, 0.80], [0.50, 0.50, 0.40]],
            site_properties={"tags": [1, 1, 0]},
        )
        slab = build_oc_slab_from_atoms(AseAtomsAdaptor.get_atoms(structure), min_ab=10.0)
        with pytest.raises(ValueError, match="No surface Pb atom"):
            PbAdsorbateSlabConfig(slab=slab, adsorbate=adsorbate, num_orientations=1)


class TestPlacedStructures:
    """The generated adslabs are consistent with the slab they came from."""

    def test_adsorbate_atoms_are_tagged_and_sit_above_the_surface(self, uniform_config, slab, adsorbate):
        adslab = uniform_config.atoms_list[0]
        assert len(adslab) == len(slab.atoms) + len(adsorbate.atoms)

        tags = np.asarray(adslab.get_tags())
        assert (tags[-len(adsorbate.atoms) :] == 2).all()
        assert (tags[: len(slab.atoms)] == np.asarray(slab.atoms.get_tags())).all()

        z = adslab.get_positions()[:, 2]
        surface_top = z[: len(slab.atoms)][np.asarray(slab.atoms.get_tags()) == 1].max()
        assert z[-len(adsorbate.atoms) :].max() > surface_top

    def test_the_slab_part_is_left_untouched(self, uniform_config, slab):
        adslab = uniform_config.atoms_list[0]
        assert np.allclose(adslab.get_positions()[: len(slab.atoms)], slab.atoms.get_positions())
        assert np.allclose(adslab.cell.array, slab.atoms.cell.array)

    def test_every_configuration_is_a_distinct_orientation(self, uniform_config, adsorbate):
        n_adsorbate = len(adsorbate.atoms)
        first = uniform_config.atoms_list[0].get_positions()[-n_adsorbate:]
        second = uniform_config.atoms_list[1].get_positions()[-n_adsorbate:]
        assert not np.allclose(first, second)

    def test_rejects_an_unknown_mode(self, slab, adsorbate):
        with pytest.raises(ValueError, match="mode must be one of"):
            PbAdsorbateSlabConfig(slab=slab, adsorbate=adsorbate, num_orientations=1, mode="not_a_mode")

    def test_the_same_seed_reproduces_the_candidate_set(self, slab, adsorbate):
        """A recorded seed is enough to regenerate a uniform run atom for atom."""
        first, second = (
            PbAdsorbateSlabConfig(
                slab=slab, adsorbate=adsorbate, num_orientations=4, mode="Pb_uniform_sample", rng=4242
            )
            for _ in range(2)
        )
        assert first.seed == 4242
        for left, right in zip(first.atoms_list, second.atoms_list):
            assert np.allclose(left.get_positions(), right.get_positions())

    def test_an_unseeded_run_still_records_the_seed_it_used(self, slab, adsorbate):
        config = PbAdsorbateSlabConfig(slab=slab, adsorbate=adsorbate, num_orientations=2, mode="Pb_uniform_sample")
        assert isinstance(config.seed, int)
        repeat = PbAdsorbateSlabConfig(
            slab=slab, adsorbate=adsorbate, num_orientations=2, mode="Pb_uniform_sample", rng=config.seed
        )
        assert np.allclose(config.atoms_list[0].get_positions(), repeat.atoms_list[0].get_positions())
