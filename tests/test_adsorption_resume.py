import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
from ase.io import read

from perovml.calculators.mock import MockCalculator
from perovml.recipes.adsorption import run_adsorption_task
from perovml.utils.structure import build_adsorbate, build_oc_slab_from_atoms

ROOT = Path(__file__).resolve().parents[1]
SLAB_FILE = ROOT / "data" / "slabs" / "FAPbI3_3x3.vasp"
MOLECULE_FILE = ROOT / "data" / "molecules" / "Acetone.vasp"


class AdsorptionResumeTest(unittest.TestCase):
    def test_run_and_resume_with_mock(self):
        calc = MockCalculator()

        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "task"

            slab_atoms = read(SLAB_FILE)
            slab = build_oc_slab_from_atoms(slab_atoms, min_ab=8.0)

            ads = build_adsorbate(molecule_file=str(MOLECULE_FILE), run_dir=task_dir)

            run_args = dict(
                task_dir=task_dir,
                slab=slab,
                adsorbate=ads,
                calc=calc,
                fmax=0.5,
                steps=2,
                pb_num_uniform=1,
                pb_num_heuristic=1,
                pb_cone=25.0,
                save_top=1,
                run_dft=False,
            )

            first = run_adsorption_task(**run_args)
            status_path = task_dir / "status.json"
            status = json.loads(status_path.read_text())

            self.assertTrue(first["mlp_done"])
            self.assertTrue(status.get("mlp_done"))
            self.assertEqual(len(status.get("configs_completed", [])), 2)

            second = run_adsorption_task(**run_args)
            status_after = json.loads(status_path.read_text())

            self.assertTrue(second["mlp_done"])
            self.assertEqual(status_after.get("configs_completed"), status.get("configs_completed"))

            # The best structure is written with the tags the DFT stage needs, in the
            # POSCAR's own atom order: a POSCAR cannot carry them itself. The Bader charge
            # transfer indexes the structure with these, so a permuted side-car would
            # attribute the charge to the wrong atoms without anything noticing.
            best_atoms = read(task_dir / "mlp" / "best.vasp")
            tags = np.array(json.loads((task_dir / "mlp" / "best_tags.json").read_text())["tags"])
            self.assertEqual(len(tags), len(best_atoms))

            symbols = np.array(best_atoms.get_chemical_symbols())
            z = best_atoms.get_positions()[:, 2]
            adsorbate = tags == 2
            self.assertEqual(int(adsorbate.sum()), len(read(MOLECULE_FILE)))
            # Acetone is C, H and O only, and it was placed above the surface layer.
            self.assertTrue(set(symbols[adsorbate]) <= {"C", "H", "O"})
            self.assertGreater(z[adsorbate].min(), z[tags == 1].max())

    @staticmethod
    def _canonical(atoms):
        """Species and positions in an order that does not depend on the file's atom order.

        A resumed run rebuilds its sampling slab from ``mlp/slab.vasp``, which is written
        with the species grouped, so the same structure can come back with its atoms
        permuted inside an element block. That permutation is not a difference in what was
        sampled, and comparing raw index by index would report one.
        """
        positions = atoms.get_positions()
        order = np.lexsort((positions[:, 2], positions[:, 1], positions[:, 0], atoms.get_atomic_numbers()))
        return atoms.get_atomic_numbers()[order], positions[order]

    def test_a_lost_candidate_archive_is_regenerated_from_the_recorded_seed(self):
        """Losing configs.traj must not quietly change what is being screened.

        The candidates are regenerated, so they have to come back as the same structures --
        which is only true because the seed that produced them is recorded in config.json.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "task"
            slab = build_oc_slab_from_atoms(read(SLAB_FILE), min_ab=8.0)
            ads = build_adsorbate(molecule_file=str(MOLECULE_FILE), run_dir=task_dir)

            run_args = dict(
                task_dir=task_dir,
                slab=slab,
                adsorbate=ads,
                calc=MockCalculator(),
                fmax=0.5,
                steps=2,
                pb_num_uniform=2,
                pb_num_heuristic=1,
                pb_cone=25.0,
                save_top=1,
                run_dft=False,
            )
            run_adsorption_task(**run_args)

            config = json.loads((task_dir / "config.json").read_text())
            self.assertIsInstance(config.get("seed"), int)

            before_z, before_xyz = self._canonical(read(task_dir / "configs" / "uniform_000.vasp"))
            (task_dir / "mlp" / "configs.traj").unlink()
            run_adsorption_task(**run_args)
            after_z, after_xyz = self._canonical(read(task_dir / "configs" / "uniform_000.vasp"))

            self.assertTrue((before_z == after_z).all())
            self.assertLess(np.abs(before_xyz - after_xyz).max(), 1e-8)

    def test_regeneration_is_refused_when_no_seed_was_recorded(self):
        """A task from before seeds were recorded must stop, not sample something new."""
        with tempfile.TemporaryDirectory() as tmpdir:
            task_dir = Path(tmpdir) / "task"
            slab = build_oc_slab_from_atoms(read(SLAB_FILE), min_ab=8.0)
            ads = build_adsorbate(molecule_file=str(MOLECULE_FILE), run_dir=task_dir)

            run_args = dict(
                task_dir=task_dir,
                slab=slab,
                adsorbate=ads,
                calc=MockCalculator(),
                fmax=0.5,
                steps=2,
                pb_num_uniform=1,
                pb_num_heuristic=1,
                pb_cone=25.0,
                save_top=1,
                run_dft=False,
            )
            run_adsorption_task(**run_args)

            # An old task directory: candidates on disk, no seed on record.
            config_path = task_dir / "config.json"
            config = json.loads(config_path.read_text())
            del config["seed"]
            config_path.write_text(json.dumps(config))
            (task_dir / "mlp" / "configs.traj").unlink()

            with self.assertRaisesRegex(RuntimeError, "records no seed"):
                run_adsorption_task(**run_args)


if __name__ == "__main__":
    unittest.main()
