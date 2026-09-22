import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from ase import Atoms
from ase.io import write

from perovml.recipes.adsorption import run_dft_stage


class FakeVASPRunner:
    def __init__(self, *args, **kwargs):
        self.calls = []

    def generate_inputs(
        self, atoms, output_dir, calc_type="scf", kpoints=None, incar_override=None, selective_dynamics=None
    ):
        self.calls.append(("gen", Path(output_dir).name, calc_type, incar_override))
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

    def run(self, input_dir, vasp_cmd="vasp_std", timeout=None, stdout_file="vasp.out", stderr_file="vasp.err"):
        self.calls.append(("run", Path(input_dir).name, vasp_cmd))
        return {"success": True, "returncode": 0}

    #: Stages whose parse must report a converged run. Anything else comes back
    #: unconverged, the way VASP leaves an OUTCAR that ran out of NELM steps.
    converged_stages = {"molecule", "slab", "adslab"}

    def parse_results(self, output_dir):
        name = Path(output_dir).name
        energy_map = {"molecule": -1.0, "slab": -2.0, "adslab": -4.0}
        converged = name in self.converged_stages
        return {
            "energy": energy_map.get(name, -0.1),
            "converged": converged,
            "error": None if converged else "electronic convergence not reached",
        }


def _setup_task_dir(tmp_path: Path) -> Path:
    task_dir = tmp_path / "task"
    mlp = task_dir / "mlp"
    mlp.mkdir(parents=True, exist_ok=True)

    mol = Atoms("H2", positions=np.array([[0, 0, 0], [0, 0, 0.74]]))
    slab = Atoms("H2", positions=np.array([[0, 0, 0], [0, 0, 1.0]]))
    adslab = Atoms("H3", positions=np.array([[0, 0, 0], [0, 0, 1.0], [0, 0, 2.0]]))

    cell = np.eye(3) * 10.0
    for a in (mol, slab, adslab):
        a.set_cell(cell)
        a.set_pbc([True, True, True])

    write(mlp / "molecule.vasp", mol, direct=True)
    write(mlp / "slab.vasp", slab, direct=True)
    write(mlp / "best.vasp", adslab, direct=True)

    return task_dir


class DFTStageTest(unittest.TestCase):
    def test_scf_mode_uses_singlepoint(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _setup_task_dir(Path(tmp))
            cfg = {"vasp_command": "vasp_std", "dft_mode": "scf"}

            with patch("perovml.dft.VASPRunner", FakeVASPRunner):
                run_dft_stage(task_dir, cfg)

            status = json.loads((task_dir / "status.json").read_text())
            self.assertTrue(status.get("dft_done"))
            self.assertEqual(status.get("dft_adsorption_energy"), -4.0 - (-2.0) - (-1.0))

    def test_opt_mode_relaxes_mol_and_adslab(self):
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _setup_task_dir(Path(tmp))
            cfg = {"vasp_command": "vasp_std", "dft_mode": "opt", "dft_incar_override": {"ENCUT": 400}}

            runner = FakeVASPRunner()
            with patch("perovml.dft.VASPRunner", lambda potcar_dir=None: runner):
                run_dft_stage(task_dir, cfg)

            status = json.loads((task_dir / "status.json").read_text())
            self.assertTrue(status.get("dft_done"))
            self.assertEqual(status.get("dft_adsorption_energy"), -4.0 - (-2.0) - (-1.0))

            # Ensure calls captured calc_type and override
            gen_calls = [c for c in runner.calls if c[0] == "gen"]
            calc_types = {name: ct for _, name, ct, _ in gen_calls}
            incar_overrides = {name: io for _, name, _, io in gen_calls}
            self.assertEqual(calc_types["molecule"], "opt")
            self.assertEqual(calc_types["adslab"], "opt")
            self.assertEqual(calc_types["slab"], "opt")
            self.assertEqual(
                incar_overrides["molecule"],
                {"ENCUT": 400, "NSW": 500, "IBRION": 2},
            )

    def test_an_unconverged_stage_stops_the_run(self):
        """An energy from an unconverged SCF must not become an adsorption energy.

        VASP exits 0 after exhausting NELM and still writes a plausible TOTEN, so nothing
        but this gate stands between a failed calculation and a published number.
        """
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _setup_task_dir(Path(tmp))
            cfg = {"vasp_command": "vasp_std", "dft_mode": "scf"}

            class UnconvergedSlab(FakeVASPRunner):
                converged_stages = {"molecule", "adslab"}

            with patch("perovml.dft.VASPRunner", UnconvergedSlab):
                run_dft_stage(task_dir, cfg)

            status = json.loads((task_dir / "status.json").read_text())
            self.assertFalse(status.get("dft_done"))
            self.assertFalse(status.get("dft_slab_done"))
            self.assertIsNone(status.get("dft_slab_energy"))
            self.assertIsNone(status.get("dft_adsorption_energy"))
            self.assertIn("convergence", status.get("dft_slab_error", ""))

    def test_a_stage_without_an_energy_stops_the_run(self):
        """A parse that found no TOTEN must not be recorded as a finished stage."""
        with tempfile.TemporaryDirectory() as tmp:
            task_dir = _setup_task_dir(Path(tmp))
            cfg = {"vasp_command": "vasp_std", "dft_mode": "scf"}

            class NoEnergy(FakeVASPRunner):
                def parse_results(self, output_dir):
                    return {"energy": None, "converged": False, "error": "no TOTEN in the OUTCAR"}

            with patch("perovml.dft.VASPRunner", NoEnergy):
                run_dft_stage(task_dir, cfg)

            status = json.loads((task_dir / "status.json").read_text())
            self.assertFalse(status.get("dft_done"))
            self.assertFalse(status.get("dft_molecule_done"))
            self.assertIn("TOTEN", status.get("dft_molecule_error", ""))


if __name__ == "__main__":
    unittest.main()
