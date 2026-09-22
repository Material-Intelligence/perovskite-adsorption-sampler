"""Tests for the VASP input files :class:`perovml.dft.vasp.VASPRunner` writes.

The four files are checked by reading them back with ``pymatgen.io.vasp.inputs``, which is
what a downstream tool would do. No VASP executable and no POTCAR directory is needed: the
POTCAR step is either skipped by expecting its error, or pointed at a directory of stub files
built in a temporary directory.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase.atoms import Atoms
from pymatgen.core import Lattice, Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

from perovml.dft.vasp import VALENCE_ELECTRONS, VASPRunner, calculate_charge_transfer, load_zvals


@pytest.fixture
def probe_atoms():
    """A six-site cell whose species are deliberately not contiguous."""
    lattice = Lattice.from_parameters(10.0, 11.0, 12.0, 90.0, 90.0, 90.0)
    structure = Structure(
        lattice,
        ["C", "O", "C", "H", "H", "O"],
        [
            [0.10, 0.10, 0.10],
            [0.20, 0.10, 0.10],
            [0.30, 0.10, 0.10],
            [0.40, 0.10, 0.10],
            [0.50, 0.10, 0.10],
            [0.60, 0.10, 0.10],
        ],
    )
    return AseAtomsAdaptor.get_atoms(structure)


@pytest.fixture
def potcar_dir(tmp_path: Path) -> Path:
    """A directory of stub POTCAR files, one per element the probe cell uses."""
    root = tmp_path / "potpaw"
    for element in ("C", "O", "H"):
        (root / element).mkdir(parents=True)
        (root / element / "POTCAR").write_text(f"STUB POTCAR {element}\n")
    return root


class TestGenerateInputs:
    """``generate_inputs`` must produce files pymatgen can read back unchanged."""

    def test_incar_round_trips(self, probe_atoms, potcar_dir, tmp_path):
        """Booleans, floats and the automatic DIPOL survive a write/read cycle."""
        runner = VASPRunner(potcar_dir=str(potcar_dir))
        out = runner.generate_inputs(probe_atoms, tmp_path / "scf", calc_type="scf", incar_override={"DIPOL": "auto"})

        incar = Incar.from_file(str(out / "INCAR"))
        assert incar["LAECHG"] is True
        assert incar["ENCUT"] == pytest.approx(500)
        assert incar["IVDW"] == 12
        # "auto" is replaced by the mean fractional coordinate, in x, y, z order.
        expected = probe_atoms.get_scaled_positions().mean(axis=0)
        assert np.allclose(incar["DIPOL"], expected, atol=1e-5)

    def test_opt_defaults_override_scf(self, probe_atoms, potcar_dir, tmp_path):
        """``calc_type="opt"`` selects the relaxation defaults, which an override can still beat."""
        runner = VASPRunner(potcar_dir=str(potcar_dir))
        out = runner.generate_inputs(probe_atoms, tmp_path / "opt", calc_type="opt", incar_override={"NSW": 42})

        incar = Incar.from_file(str(out / "INCAR"))
        assert incar["IBRION"] == 2
        assert incar["NSW"] == 42
        assert incar["LAECHG"] is False

    def test_kpoints_are_gamma_centred(self, probe_atoms, potcar_dir, tmp_path):
        """The requested mesh is written as a Gamma-centred automatic grid."""
        runner = VASPRunner(potcar_dir=str(potcar_dir))
        out = runner.generate_inputs(probe_atoms, tmp_path / "kpts", kpoints=(2, 3, 1))

        kpoints = Kpoints.from_file(str(out / "KPOINTS"))
        assert kpoints.style == Kpoints.supported_modes.Gamma
        assert tuple(kpoints.kpts[0]) == (2, 3, 1)

    def test_partial_selective_dynamics_survive(self, probe_atoms, potcar_dir, tmp_path):
        """A ``T T F`` site stays ``T T F``, which a FixAtoms round-trip cannot manage."""
        flags = np.zeros((len(probe_atoms), 3), dtype=bool)
        flags[0] = (True, True, False)
        flags[1] = (True, True, True)

        runner = VASPRunner(potcar_dir=str(potcar_dir))
        out = runner.generate_inputs(probe_atoms, tmp_path / "sd", selective_dynamics=flags)

        poscar = Poscar.from_file(str(out / "POSCAR"))
        assert np.array_equal(np.array(poscar.selective_dynamics), flags)

    def test_no_selective_dynamics_block_when_everything_is_free(self, probe_atoms, potcar_dir, tmp_path):
        """An all-True array means no constraints, so VASP gets no block at all."""
        flags = np.ones((len(probe_atoms), 3), dtype=bool)

        runner = VASPRunner(potcar_dir=str(potcar_dir))
        out = runner.generate_inputs(probe_atoms, tmp_path / "free", selective_dynamics=flags)

        assert "Selective dynamics" not in (out / "POSCAR").read_text()
        assert Poscar.from_file(str(out / "POSCAR")).selective_dynamics is None

    def test_poscar_keeps_the_structure(self, probe_atoms, potcar_dir, tmp_path):
        """Lattice, species order and fractional coordinates all come back unchanged."""
        runner = VASPRunner(potcar_dir=str(potcar_dir))
        out = runner.generate_inputs(probe_atoms, tmp_path / "poscar")

        structure = Poscar.from_file(str(out / "POSCAR")).structure
        assert np.allclose(structure.lattice.matrix, np.array(probe_atoms.get_cell()))
        assert [site.specie.symbol for site in structure] == probe_atoms.get_chemical_symbols()
        assert np.allclose(structure.frac_coords, probe_atoms.get_scaled_positions(), atol=1e-6)

    def test_potcar_blocks_match_the_poscar(self, probe_atoms, potcar_dir, tmp_path):
        """One POTCAR block per POSCAR species block, in the same order.

        The probe cell is C O C H H O, so the POSCAR has five blocks and two of them are
        carbon. A POTCAR built from the unique elements instead would have three, and VASP
        would read the structure wrongly without complaining.
        """
        runner = VASPRunner(potcar_dir=str(potcar_dir))
        out = runner.generate_inputs(probe_atoms, tmp_path / "potcar")

        site_symbols = Poscar.from_file(str(out / "POSCAR")).site_symbols
        assert site_symbols == ["C", "O", "C", "H", "O"]

        blocks = [line.split()[-1] for line in (out / "POTCAR").read_text().splitlines() if line.strip()]
        assert blocks == site_symbols

    def test_missing_potcar_dir_is_an_error(self, probe_atoms, tmp_path):
        """Without a POTCAR directory the runner refuses rather than writing three of four files."""
        runner = VASPRunner(potcar_dir="")
        with pytest.raises(ValueError, match="POTCAR directory not set"):
            runner.generate_inputs(probe_atoms, tmp_path / "nopotcar")


class TestPotcarValences:
    """The valences used for charge transfer come from the POTCARs that were written."""

    def test_zval_is_read_from_the_potcar_that_was_used(self, tmp_path):
        """A ``Pb_d`` POTCAR contributes 14 electrons, not the 4 of the plain ``Pb``.

        Getting this wrong is worth ten electrons per Pb atom in every Bader charge
        transfer, and nothing downstream can notice.
        """
        root = tmp_path / "potpaw"
        (root / "Pb_d").mkdir(parents=True)
        (root / "Pb_d" / "POTCAR").write_text(
            "  PAW_PBE Pb_d 06Sep2000\n  14.0000000000000\n   POMASS =  207.200; ZVAL   =  14.000    mass and valenz\n"
        )
        (root / "I").mkdir(parents=True)
        (root / "I" / "POTCAR").write_text(
            "  PAW_PBE I 08Apr2002\n   7.0000000000000\n   POMASS =  126.904; ZVAL   =   7.000    mass and valenz\n"
        )

        lattice = Lattice.from_parameters(8.0, 8.0, 8.0, 90.0, 90.0, 90.0)
        atoms = AseAtomsAdaptor.get_atoms(Structure(lattice, ["Pb", "I"], [[0, 0, 0], [0.5, 0.5, 0.5]]))

        runner = VASPRunner(potcar_dir=str(root))
        out = runner.generate_inputs(atoms, tmp_path / "pbi")

        valences = load_zvals(out)
        assert valences == {"Pb": 14.0, "I": 7.0}
        # The fallback table, which assumes a different variant, disagrees by ten.
        assert VALENCE_ELECTRONS["Pb"] == 4

    def test_charge_transfer_uses_the_supplied_valences(self):
        """The reference count follows the POTCAR mapping, not the fallback table."""
        atoms = Atoms("PbN", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]], cell=np.eye(3) * 10.0)
        bader_charges = np.array([12.0, 5.4])

        result = calculate_charge_transfer(bader_charges, atoms, [1], valence_electrons={"Pb": 14.0, "N": 5.0})
        assert result["reference_electrons"] == pytest.approx(5.0)
        assert result["charge_transfer"] == pytest.approx(0.4)
        assert result["assumed_elements"] == []

    def test_a_missing_valence_falls_back_and_says_so(self):
        """A guessed valence is still reported, so it cannot pass for a measured one."""
        atoms = Atoms("PbN", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]], cell=np.eye(3) * 10.0)
        result = calculate_charge_transfer(np.array([12.0, 5.4]), atoms, [1], valence_electrons={})
        assert result["assumed_elements"] == ["N"]

    def test_an_unknown_element_is_an_error(self):
        atoms = Atoms("PbXe", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 2.5]], cell=np.eye(3) * 10.0)
        with pytest.raises(ValueError, match="no valence count for Xe"):
            calculate_charge_transfer(np.array([12.0, 7.9]), atoms, [1], valence_electrons={})


class TestParseResults:
    """``parse_results`` decides whether an energy may be used at all."""

    @staticmethod
    def _write_outcar(
        directory: Path,
        *,
        scf_steps: int,
        nelm: int = 60,
        nsw: int = 0,
        ibrion: int = -1,
        reached_accuracy: bool = False,
        energy: float = -123.456,
    ) -> Path:
        """Write the handful of OUTCAR lines the parser looks at."""
        directory.mkdir(parents=True, exist_ok=True)
        lines = [
            f"   NELM   =    {nelm};   NELMIN=  2; NELMDL= -5     number of ELM steps",
            f"   NSW    =      {nsw}    number of steps for IOM",
            f"   IBRION =     {ibrion}    ionic relax: 0-MD 1-quasi-New 2-CG",
        ]
        for step in range(1, scf_steps + 1):
            lines.append(f"DAV:   {step}    -0.123456789012E+03   -0.11406E+04   -0.35917E+04  4560   0.113E+03")
        lines.append(f"  free  energy   TOTEN  =     {energy:.6f} eV")
        lines.append("  LOOP+:  cpu time   12.3456: real time   12.3456")
        if reached_accuracy:
            lines.append(" reached required accuracy - stopping structural energy minimisation")
        (directory / "OUTCAR").write_text("\n".join(lines) + "\n")
        return directory

    def test_a_converged_single_point_is_accepted(self, tmp_path):
        out = self._write_outcar(tmp_path / "scf", scf_steps=11)
        parsed = VASPRunner().parse_results(out)

        assert parsed["energy"] == pytest.approx(-123.456)
        assert parsed["converged"] is True
        assert parsed["electronic_converged"] is True
        assert parsed["ionic_converged"] is True
        assert parsed["n_electronic_steps_last_cycle"] == 11
        assert parsed["error"] is None

    def test_an_scf_that_exhausted_nelm_is_rejected(self, tmp_path):
        """The energy is present and looks ordinary; only the step count gives it away."""
        out = self._write_outcar(tmp_path / "nelm", scf_steps=60, nelm=60)
        parsed = VASPRunner().parse_results(out)

        assert parsed["energy"] == pytest.approx(-123.456)
        assert parsed["converged"] is False
        assert parsed["electronic_converged"] is False
        assert "NELM" in parsed["error"]

    def test_a_relaxation_needs_the_accuracy_marker(self, tmp_path):
        out = self._write_outcar(tmp_path / "opt", scf_steps=8, nsw=200, ibrion=2)
        parsed = VASPRunner().parse_results(out)
        assert parsed["electronic_converged"] is True
        assert parsed["ionic_converged"] is False
        assert parsed["converged"] is False
        assert "ionic convergence" in parsed["error"]

        done = self._write_outcar(tmp_path / "opt_done", scf_steps=8, nsw=200, ibrion=2, reached_accuracy=True)
        assert VASPRunner().parse_results(done)["converged"] is True

    def test_a_fortran_d_exponent_and_loose_spacing_still_parse(self, tmp_path):
        """The TOTEN spelling varies between VASP versions; the number must still be read."""
        directory = tmp_path / "spacing"
        directory.mkdir()
        (directory / "OUTCAR").write_text(
            "   NELM   =    60;   NELMIN=  2; NELMDL= -5\n"
            "   NSW    =      0\n"
            "   IBRION =     -1\n"
            "DAV:   3    -0.1E+03   -0.1E+04   -0.3E+04  4560   0.1E+03\n"
            "  free energy    TOTEN  =    -0.98765432D+02 eV\n"
            "  LOOP+:  cpu time   1.0: real time   1.0\n"
        )
        parsed = VASPRunner().parse_results(directory)
        assert parsed["energy"] == pytest.approx(-98.765432)
        assert parsed["converged"] is True

    def test_an_outcar_without_an_energy_is_rejected(self, tmp_path):
        directory = tmp_path / "empty"
        directory.mkdir()
        (directory / "OUTCAR").write_text("   NELM   =    60\n   NSW    =      0\n   IBRION =     -1\n")
        parsed = VASPRunner().parse_results(directory)
        assert parsed["energy"] is None
        assert parsed["converged"] is False
        assert "TOTEN" in parsed["error"]

    def test_a_missing_outcar_is_reported(self, tmp_path):
        parsed = VASPRunner().parse_results(tmp_path / "nothing")
        assert parsed["converged"] is False
        assert parsed["error"] == "OUTCAR not found"
