"""VASP execution and Bader charge analysis.

The module provides:

  - :class:`VASPRunner`, which writes the four VASP input files, runs the code and
    parses the output;
  - :func:`run_bader`, which drives the Bader executable on a CHGCAR;
  - :func:`calculate_charge_transfer`, which turns Bader charges into the electron
    transfer onto an adsorbate.

Notes:
    The POTCAR files are never shipped with this repository. Point ``potcar_dir``
    at your own PAW directory, or export ``VASP_PP_PATH``.

    INCAR, POSCAR and KPOINTS are written through ``pymatgen.io.vasp.inputs``, so
    the files this module produces are exactly what pymatgen would read back. The
    POTCAR is assembled here rather than through ``pymatgen.io.vasp.Potcar``,
    because the potentials are looked up in a user-supplied directory that carries
    no pymatgen ``functional`` label. The OUTCAR parsing is likewise hand-rolled:
    it reads the handful of quantities this workflow needs from a text scan,
    without requiring a complete, converged run the way ``Vasprun`` does.
"""

from __future__ import annotations

import json
import logging
import os
import re
import subprocess
from pathlib import Path
from typing import Any, Literal

import numpy as np
from ase.atoms import Atoms
from ase.io import read
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp.inputs import Incar, Kpoints, Poscar

logger = logging.getLogger(__name__)

# OUTCAR patterns. The spacing inside 'free  energy   TOTEN' varies between VASP
# versions, so it is matched loosely; the number may carry a Fortran D exponent.
_TOTEN_RE = re.compile(r"free\s+energy\s+TOTEN\s*=\s*([-+0-9.EDed]+)")
_FMAX_RE = re.compile(r"FORCES:\s*max\s*=\s*([-+0-9.EDed]+)")
#: One SCF iteration line, e.g. 'DAV:   3    -0.28E+03 ...'. The number is the iteration.
_SCF_STEP_RE = re.compile(r"^\s*(?:DAV|RMM|EDDAV|CG|DMP|SDA|JDH|CHI)\s*:\s*(\d+)")
_NELM_RE = re.compile(r"NELM\s*=\s*(\d+)")
_NSW_RE = re.compile(r"NSW\s*=\s*(\d+)")
_IBRION_RE = re.compile(r"IBRION\s*=\s*(-?\d+)")


def _to_float(text: str) -> float:
    """Parse a Fortran-formatted real, accepting a ``D`` exponent.

    Args:
        text: The number as VASP wrote it, e.g. ``-0.123E+03`` or ``-0.123D+03``.

    Returns:
        The value as a Python float.
    """
    return float(text.replace("D", "E").replace("d", "e"))


__all__ = [
    "DEFAULT_INCAR_OPT",
    "DEFAULT_INCAR_SCF",
    "VALENCE_ELECTRONS",
    "ZVAL_FILENAME",
    "VASPRunner",
    "calculate_charge_transfer",
    "combine_aeccar",
    "load_zvals",
    "run_bader",
]


# =============================================================================
# Valence electrons for common elements (used in charge transfer calculation)
# =============================================================================

ZVAL_FILENAME = "zval.json"
"""File :meth:`VASPRunner.generate_inputs` writes the POTCAR valences into."""

_ZVAL_RE = re.compile(r"ZVAL\s*=\s*([0-9.]+)")

VALENCE_ELECTRONS: dict[str, int] = {
    # First row
    "H": 1,
    # Second row
    "C": 4,
    "N": 5,
    "O": 6,
    "F": 7,
    # Third row
    "Si": 4,
    "P": 5,
    "S": 6,
    "Cl": 7,
    # Fourth row (selected)
    "Br": 7,
    "I": 7,
    # Metals (common in perovskites)
    "Pb": 4,  # the plain Pb potential; Pb_d, which _write_potcar prefers, carries 14
    "Sn": 4,
    "Cs": 9,  # Cs_sv
    "Rb": 9,
    "K": 9,  # K_sv
    "Na": 7,  # Na_pv
}
"""Fallback valence counts, one guess per element.

Each entry assumes one particular POTCAR variant, and the variant
:meth:`VASPRunner._write_potcar` actually picks may be another one -- ``Pb_d`` carries 14
valence electrons where the plain ``Pb`` carries 4, which is 10 electrons per Pb atom of
error in a charge transfer. Use the ``zval.json`` written next to the VASP inputs instead;
this table is only the last resort, and using it logs a warning.
"""


# =============================================================================
# Default INCAR parameters
# =============================================================================

DEFAULT_INCAR_SCF: dict[str, Any] = {
    # General
    "PREC": "Accurate",
    "ENCUT": 500,
    "EDIFF": 1e-6,
    "NELM": 200,
    # Electronic
    "ISMEAR": 0,
    "SIGMA": 0.05,
    # DFT-D3(BJ) dispersion correction
    "IVDW": 12,
    # Output for Bader analysis
    "LCHARG": True,
    "LAECHG": True,  # all-electron charge density, needed for Bader
}
"""INCAR defaults for a single-point calculation.

No parallel layout is set here. ``NCORE`` has to divide the cores per node, and a value
that does not either aborts the run or quietly costs a large factor in speed, so a number
picked in this file would be wrong on most machines -- and nothing under ``src/perovml/``
is supposed to know about your cluster. VASP's own defaults are safe; set ``NCORE`` and
``KPAR`` for your node through ``dft_incar_override`` in the workflow config.
"""

DEFAULT_INCAR_OPT: dict[str, Any] = {
    **DEFAULT_INCAR_SCF,
    # Optimization settings
    "IBRION": 2,  # conjugate gradient
    "NSW": 100,
    "EDIFFG": -0.03,  # force convergence, eV/A
    "ISIF": 2,  # relax the ions only
    # Reduce output for optimization
    "LCHARG": False,
    "LAECHG": False,
}
"""INCAR defaults for an ionic relaxation."""


# =============================================================================
# VASPRunner
# =============================================================================


class VASPRunner:
    """Write VASP inputs, run the code and parse its output.

    Examples:
        >>> runner = VASPRunner(potcar_dir="/path/to/potpaw_PBE")  # doctest: +SKIP
        >>> runner.generate_inputs(atoms, "dft/molecule", calc_type="scf")  # doctest: +SKIP
        >>> runner.run("dft/molecule", vasp_cmd="srun vasp_std")  # doctest: +SKIP
        >>> results = runner.parse_results("dft/molecule")  # doctest: +SKIP

    Attributes:
        potcar_dir: Directory holding one sub-directory of PAW potentials per element.
        default_kpoints: K-point grid used when a call does not specify one.
    """

    def __init__(
        self,
        potcar_dir: str | None = None,
        default_kpoints: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        """Initialise the runner.

        Args:
            potcar_dir: Path to the VASP POTCAR directory, e.g. a ``potpaw_PBE``
                tree. Falls back to the ``VASP_PP_PATH`` environment variable.
            default_kpoints: Default k-point grid for molecules and slabs.
        """
        self.potcar_dir = potcar_dir or os.environ.get("VASP_PP_PATH", "")
        self.default_kpoints = default_kpoints

    def generate_inputs(
        self,
        atoms: Atoms,
        output_dir: str | Path,
        calc_type: Literal["scf", "opt"] = "scf",
        kpoints: tuple[int, int, int] | None = None,
        incar_override: dict[str, Any] | None = None,
        selective_dynamics: np.ndarray | None = None,
    ) -> Path:
        """Write INCAR, POSCAR, KPOINTS and POTCAR into a directory.

        Setting ``DIPOL`` to the string ``"auto"`` in ``incar_override`` replaces it
        with the mean fractional coordinate of the structure.

        Args:
            atoms: Structure to write.
            output_dir: Directory for the input files. Created if missing.
            calc_type: ``"scf"`` for a single point, ``"opt"`` for a relaxation.
            kpoints: K-point grid. Defaults to :attr:`default_kpoints`.
            incar_override: INCAR entries layered on top of the defaults.
            selective_dynamics: Boolean array of shape ``(N, 3)``, in site order, where
                True means the site is free along that Cartesian direction. Written
                verbatim, so partly constrained sites such as ``T T F`` survive. When
                omitted, an ASE ``FixAtoms`` constraint on ``atoms`` is used instead.

        Returns:
            The directory that was written to.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # INCAR
        incar = DEFAULT_INCAR_OPT.copy() if calc_type == "opt" else DEFAULT_INCAR_SCF.copy()
        if incar_override:
            incar.update(incar_override)
        # DIPOL: "auto" means the mean fractional coordinate of the structure.
        if isinstance(incar.get("DIPOL"), str) and incar["DIPOL"].lower() == "auto":
            frac_mean = atoms.get_scaled_positions().mean(axis=0)
            incar["DIPOL"] = f"{frac_mean[0]:.6f} {frac_mean[1]:.6f} {frac_mean[2]:.6f}"
        self._write_incar(output_dir / "INCAR", incar)

        # POSCAR
        poscar = self._build_poscar(atoms, selective_dynamics)
        poscar.write_file(str(output_dir / "POSCAR"))

        # KPOINTS
        self._write_kpoints(output_dir / "KPOINTS", kpoints or self.default_kpoints)

        # POTCAR, in the species order the POSCAR just used. The valences come back from
        # the files that were actually concatenated, and are stored next to them so that a
        # later Bader analysis does not have to guess which variant was used.
        zvals = self._write_potcar(output_dir / "POTCAR", poscar.site_symbols)
        with open(output_dir / ZVAL_FILENAME, "w") as file:
            json.dump(zvals, file, indent=2, sort_keys=True)

        return output_dir

    def run(
        self,
        input_dir: str | Path,
        vasp_cmd: str = "vasp_std",
        timeout: int | None = None,
        stdout_file: str = "vasp.out",
        stderr_file: str = "vasp.err",
    ) -> dict[str, Any]:
        """Run VASP in a prepared input directory.

        Args:
            input_dir: Directory holding the VASP input files.
            vasp_cmd: Command to run, e.g. ``"vasp_std"`` or ``"srun vasp_std"``.
                It is split on whitespace, not passed through a shell.
            timeout: Maximum runtime in seconds, or None for no limit.
            stdout_file: File in ``input_dir`` that stdout is redirected to.
            stderr_file: File in ``input_dir`` that stderr is redirected to.

        Returns:
            A dict with ``success``, ``returncode`` and ``error``.
        """
        input_dir = Path(input_dir)

        with open(input_dir / stdout_file, "w") as stdout, open(input_dir / stderr_file, "w") as stderr:
            try:
                result = subprocess.run(
                    vasp_cmd.split(),
                    cwd=input_dir,
                    stdout=stdout,
                    stderr=stderr,
                    timeout=timeout,
                )
                return {
                    "success": result.returncode == 0,
                    "returncode": result.returncode,
                    "error": None,
                }
            except subprocess.TimeoutExpired:
                return {
                    "success": False,
                    "returncode": -1,
                    "error": f"Timeout after {timeout}s",
                }
            except OSError as exc:
                return {
                    "success": False,
                    "returncode": -1,
                    "error": str(exc),
                }

    def parse_results(self, output_dir: str | Path) -> dict[str, Any]:
        """Parse the OUTCAR, and the CONTCAR when it is present.

        The OUTCAR is streamed line by line in a single pass: a relaxation OUTCAR runs to
        hundreds of megabytes, and this is called once per stage per molecule.

        ``converged`` is the answer to "may this energy be used?", not just "did VASP
        print a happy word". It requires both:

          - **electronic convergence**: the last SCF cycle finished in fewer than ``NELM``
            iterations. A run that exhausts NELM still writes a perfectly normal TOTEN and
            still exits 0, so this is the only thing standing between an unconverged SCF
            and a published adsorption energy;
          - **ionic convergence**: for a relaxation (``NSW > 0`` and ``IBRION >= 0``), the
            ``reached required accuracy`` marker. A single point has nothing to converge
            ionically and passes this part by definition.

        Args:
            output_dir: Directory holding the VASP output files.

        Returns:
            A dict with ``energy`` (the last free energy TOTEN, in eV), ``converged``,
            ``electronic_converged``, ``ionic_converged``, ``n_ionic_steps``,
            ``n_electronic_steps``, ``n_electronic_steps_last_cycle``, ``nelm`` and
            ``error``, plus ``fmax`` and ``atoms`` when those could be recovered.
            ``error`` names the reason whenever ``converged`` is False.
        """
        output_dir = Path(output_dir)
        results: dict[str, Any] = {
            "energy": None,
            "forces": None,
            "converged": False,
            "electronic_converged": False,
            "ionic_converged": False,
            "n_ionic_steps": 0,
            "n_electronic_steps": 0,
            "n_electronic_steps_last_cycle": 0,
            "nelm": None,
            "error": None,
        }

        outcar_path = output_dir / "OUTCAR"
        if not outcar_path.exists():
            results["error"] = "OUTCAR not found"
            return results

        nelm: int | None = None
        nsw: int | None = None
        ibrion: int | None = None
        energy: float | None = None
        fmax: float | None = None
        n_ionic = 0
        n_electronic = 0
        steps_this_cycle = 0
        steps_last_cycle = 0
        reached_accuracy = False

        try:
            with open(outcar_path) as file:
                for line in file:
                    if "TOTEN" in line:
                        match = _TOTEN_RE.search(line)
                        if match:
                            energy = _to_float(match.group(1))
                        continue
                    if "LOOP+:" in line:
                        n_ionic += 1
                        steps_last_cycle = steps_this_cycle
                        steps_this_cycle = 0
                        continue
                    scf = _SCF_STEP_RE.match(line)
                    if scf:
                        n_electronic += 1
                        steps_this_cycle = int(scf.group(1))
                        continue
                    if "FORCES:" in line:
                        match = _FMAX_RE.search(line)
                        if match:
                            fmax = _to_float(match.group(1))
                        continue
                    if "reached required accuracy" in line:
                        reached_accuracy = True
                        continue
                    if nelm is None and "NELM" in line:
                        match = _NELM_RE.search(line)
                        if match:
                            nelm = int(match.group(1))
                    if nsw is None and "NSW" in line:
                        match = _NSW_RE.search(line)
                        if match:
                            nsw = int(match.group(1))
                    if ibrion is None and "IBRION" in line:
                        match = _IBRION_RE.search(line)
                        if match:
                            ibrion = int(match.group(1))
        except (OSError, ValueError) as exc:
            results["error"] = str(exc)
            return results

        # An ionic cycle that VASP never closed with a LOOP+ line, e.g. a run killed
        # mid-SCF, still tells us how far the last cycle got.
        if steps_this_cycle:
            steps_last_cycle = steps_this_cycle

        results["energy"] = energy
        if fmax is not None:
            results["fmax"] = fmax
        results["n_ionic_steps"] = n_ionic
        results["n_electronic_steps"] = n_electronic
        results["n_electronic_steps_last_cycle"] = steps_last_cycle
        results["nelm"] = nelm

        # Relaxed structure, when VASP got far enough to write a CONTCAR.
        contcar_path = output_dir / "CONTCAR"
        if contcar_path.exists() and contcar_path.stat().st_size > 0:
            try:
                results["atoms"] = read(contcar_path)
            except (OSError, ValueError, IndexError):
                logger.debug("could not read %s", contcar_path, exc_info=True)

        problems: list[str] = []
        if energy is None:
            problems.append("no 'free energy TOTEN' line in the OUTCAR")

        # Electronic convergence.
        if nelm is None or steps_last_cycle == 0:
            problems.append("could not determine electronic convergence (no NELM or no SCF cycle in the OUTCAR)")
        elif steps_last_cycle >= nelm:
            problems.append(f"electronic convergence not reached: the last SCF cycle used all {nelm} NELM steps")
        else:
            results["electronic_converged"] = True

        # Ionic convergence: only a relaxation has something to reach.
        is_relaxation = bool(nsw and nsw > 0 and (ibrion is None or ibrion >= 0))
        if not is_relaxation:
            results["ionic_converged"] = True
        elif reached_accuracy:
            results["ionic_converged"] = True
        else:
            problems.append(f"ionic convergence not reached in {n_ionic} of {nsw} steps")

        results["converged"] = results["electronic_converged"] and results["ionic_converged"] and energy is not None
        if problems:
            results["error"] = "; ".join(problems)

        return results

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _write_incar(path: Path, params: dict[str, Any]) -> None:
        """Write an INCAR file through :class:`pymatgen.io.vasp.inputs.Incar`.

        Args:
            path: File to write.
            params: INCAR tags. ``Incar`` normalises the values, so a Python bool
                lands as a Fortran logical and a vector tag such as ``DIPOL`` may be
                given either as a list or as a space-separated string.
        """
        Incar(params).write_file(path)

    @staticmethod
    def _build_poscar(atoms: Atoms, selective_dynamics: np.ndarray | None = None) -> Poscar:
        """Turn an ASE structure into a :class:`pymatgen.io.vasp.inputs.Poscar`.

        The conversion goes through :class:`~pymatgen.io.ase.AseAtomsAdaptor`, which
        also carries an ASE ``FixAtoms`` constraint over as selective dynamics. An
        explicit ``selective_dynamics`` argument overrides whatever the constraints
        imply, and unlike a ``FixAtoms`` round-trip it keeps per-axis flags such as
        ``T T F`` intact.

        Args:
            atoms: Structure to write. It must carry a non-degenerate cell, since a
                POSCAR has no way to express one that does not.
            selective_dynamics: Boolean array of shape ``(N, 3)``, in site order.
                When every flag is True the block is omitted, as VASP expects.

        Returns:
            The ``Poscar`` object, ready to write or to inspect.
        """
        structure = AseAtomsAdaptor.get_structure(atoms)
        flags = None if selective_dynamics is None else np.asarray(selective_dynamics, dtype=bool).tolist()
        return Poscar(structure, selective_dynamics=flags)

    @staticmethod
    def _write_kpoints(path: Path, kpoints: tuple[int, int, int]) -> None:
        """Write a Gamma-centred automatic KPOINTS mesh.

        Args:
            path: File to write.
            kpoints: Subdivisions along the three reciprocal lattice vectors.
        """
        Kpoints.gamma_automatic(kpts=tuple(int(k) for k in kpoints)).write_file(str(path))

    def _write_potcar(self, path: Path, site_symbols: list[str]) -> dict[str, dict[str, Any]]:
        """Concatenate one POTCAR per POSCAR species block, in POSCAR order.

        Potentials are looked up as ``<potcar_dir>/<variant>/POTCAR``, trying the
        semi-core variants first where they exist.

        Args:
            path: File to write.
            site_symbols: Element symbols in POSCAR block order, as
                :attr:`pymatgen.io.vasp.inputs.Poscar.site_symbols` reports them. Taking
                them from the POSCAR rather than recomputing them is what keeps the two
                files in step when one element occupies more than one block.

        Returns:
            Per element, the ``variant`` that was used and its ``zval``, read out of the
            file itself. ``zval`` is None when the file carries no ``ZVAL`` field.

        Raises:
            ValueError: If no POTCAR directory was configured.
            FileNotFoundError: If no variant of an element's POTCAR was found.
        """
        if not self.potcar_dir:
            raise ValueError(
                "POTCAR directory not set. Set the VASP_PP_PATH environment variable "
                "or pass potcar_dir to VASPRunner."
            )

        unique_elements: list[str] = list(site_symbols)

        # Preferred POTCAR variants, semi-core states first where they help accuracy.
        potcar_variants = {
            "Pb": ["Pb_d", "Pb"],
            "I": ["I", "I_sv"],
            "Cs": ["Cs_sv", "Cs"],
            "K": ["K_sv", "K_pv", "K"],
            "Na": ["Na_pv", "Na_sv", "Na"],
            "C": ["C", "C_s"],
            "N": ["N", "N_s"],
            "O": ["O", "O_s"],
            "H": ["H", "H_s"],
            "S": ["S", "S_h"],
            "F": ["F", "F_s"],
            "Cl": ["Cl", "Cl_h"],
        }

        potcar_content: list[str] = []
        zvals: dict[str, dict[str, Any]] = {}
        for element in unique_elements:
            variants = potcar_variants.get(element, [element])
            for variant in variants:
                potcar_path = Path(self.potcar_dir) / variant / "POTCAR"
                if potcar_path.exists():
                    with open(potcar_path) as file:
                        content = file.read()
                    potcar_content.append(content)
                    match = _ZVAL_RE.search(content)
                    zvals[element] = {
                        "variant": variant,
                        "zval": float(match.group(1)) if match else None,
                    }
                    if match is None:
                        logger.warning("no ZVAL field in %s; charge transfer will need a fallback", potcar_path)
                    break
            else:
                raise FileNotFoundError(f"POTCAR for {element} not found in {self.potcar_dir}. Tried: {variants}")

        with open(path, "w") as file:
            file.write("".join(potcar_content))

        return zvals


# =============================================================================
# Bader analysis
# =============================================================================


def run_bader(
    chgcar_path: str | Path,
    aeccar_path: str | Path | None = None,
    bader_cmd: str = "bader",
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run a Bader charge analysis on a CHGCAR.

    Args:
        chgcar_path: Path to the CHGCAR.
        aeccar_path: Path to a combined AECCAR (AECCAR0 + AECCAR2) to use as the
            reference density, which is more accurate. None uses the CHGCAR alone.
        bader_cmd: Name of the Bader executable.
        output_dir: Directory the analysis runs in. Defaults to the CHGCAR's directory.

    Returns:
        A dict with ``charges`` and ``volumes`` (both arrays), ``success`` and ``error``.
    """
    chgcar_path = Path(chgcar_path)
    output_dir = Path(output_dir) if output_dir else chgcar_path.parent

    results: dict[str, Any] = {
        "charges": None,
        "volumes": None,
        "success": False,
        "error": None,
    }

    cmd = [bader_cmd, str(chgcar_path)]
    if aeccar_path:
        cmd.extend(["-ref", str(aeccar_path)])

    try:
        subprocess.run(cmd, cwd=output_dir, capture_output=True, check=True)

        acf_path = output_dir / "ACF.dat"
        if acf_path.exists():
            charges, volumes = _parse_acf_dat(acf_path)
            results["charges"] = charges
            results["volumes"] = volumes
            results["success"] = True
        else:
            results["error"] = "ACF.dat not found after Bader analysis"

    except subprocess.CalledProcessError as exc:
        results["error"] = f"Bader failed: {exc.stderr.decode()}"
    except FileNotFoundError:
        results["error"] = f"Bader executable '{bader_cmd}' not found"
    except OSError as exc:
        results["error"] = str(exc)

    return results


def _parse_acf_dat(acf_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Parse the ``ACF.dat`` table the Bader code writes.

    Args:
        acf_path: Path to ``ACF.dat``.

    Returns:
        A tuple of the per-atom Bader electron counts and the per-atom volumes.
        The volume array is empty when the file does not carry that column.
    """
    charges: list[float] = []
    volumes: list[float] = []

    with open(acf_path) as file:
        for raw_line in file:
            line = raw_line.strip()
            if not line or line.startswith("#") or line.startswith("-"):
                continue
            parts = line.split()
            # Columns: index, X, Y, Z, CHARGE, MIN_DIST, ATOMIC_VOL
            if len(parts) >= 5 and parts[0].isdigit():
                charges.append(float(parts[4]))
                if len(parts) >= 7:
                    volumes.append(float(parts[6]))

    return np.array(charges), np.array(volumes) if volumes else np.array([])


def load_zvals(input_dir: str | Path) -> dict[str, float]:
    """Read the POTCAR valences :meth:`VASPRunner.generate_inputs` recorded.

    Args:
        input_dir: Directory holding the VASP inputs, i.e. the one that has
            :data:`ZVAL_FILENAME` in it.

    Returns:
        Element to valence-electron count, empty when the file is missing, unreadable, or
        carries no usable value.
    """
    zval_path = Path(input_dir) / ZVAL_FILENAME
    if not zval_path.exists():
        return {}
    try:
        with open(zval_path) as file:
            raw = json.load(file)
    except (OSError, ValueError):
        logger.warning("could not read %s", zval_path, exc_info=True)
        return {}

    valences: dict[str, float] = {}
    for element, record in raw.items():
        zval = record.get("zval") if isinstance(record, dict) else record
        if zval is not None:
            valences[element] = float(zval)
    return valences


def calculate_charge_transfer(
    bader_charges: np.ndarray,
    atoms: Atoms,
    adsorbate_indices: list[int],
    valence_electrons: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Turn Bader charges into the electron transfer onto an adsorbate.

    The transfer is the Bader electron count on the adsorbate minus the electron
    count its free atoms would carry. A positive value means the adsorbate gained
    electrons (reduction), a negative value that it lost them (oxidation).

    The reference count has to come from the POTCAR that was actually used: the same
    element has different valences in different PAW variants. Pass the mapping
    :func:`load_zvals` reads back from the calculation directory. Falling back to
    :data:`VALENCE_ELECTRONS` logs a warning for every element it has to guess.

    Args:
        bader_charges: Per-atom Bader electron counts, as returned by :func:`run_bader`.
        atoms: Structure in the same atom order as the Bader calculation.
        adsorbate_indices: Indices of the adsorbate atoms within ``atoms``.
        valence_electrons: Element to valence-electron mapping, normally from
            :func:`load_zvals`. Missing elements fall back to :data:`VALENCE_ELECTRONS`.

    Returns:
        A dict with the total ``charge_transfer``, the ``bader_electrons`` and
        ``reference_electrons`` it was computed from, the per-atom breakdown, and
        ``assumed_elements``: the elements whose reference count came from the fallback
        table rather than from a POTCAR.

    Raises:
        ValueError: If an adsorbate element is in neither the supplied mapping nor the
            fallback table.
    """
    valence = dict(valence_electrons or {})
    symbols = atoms.get_chemical_symbols()

    assumed: list[str] = []
    ref_electrons = 0.0
    per_atom_reference: list[float] = []
    for index in adsorbate_indices:
        element = symbols[index]
        if element not in valence:
            if element not in VALENCE_ELECTRONS:
                raise ValueError(
                    f"no valence count for {element}: it is neither in the POTCAR valences "
                    f"({sorted(valence)}) nor in VALENCE_ELECTRONS."
                )
            valence[element] = float(VALENCE_ELECTRONS[element])
            if element not in assumed:
                assumed.append(element)
        ref_electrons += valence[element]
        per_atom_reference.append(valence[element])

    if assumed:
        logger.warning(
            "no POTCAR valence for %s; assuming %s from the fallback table, which is tied to one "
            "PAW variant and may be wrong",
            ", ".join(assumed),
            {element: VALENCE_ELECTRONS[element] for element in assumed},
        )

    bader_electrons = sum(bader_charges[index] for index in adsorbate_indices)
    charge_transfer = bader_electrons - ref_electrons

    return {
        "charge_transfer": float(charge_transfer),
        "bader_electrons": float(bader_electrons),
        "reference_electrons": float(ref_electrons),
        "adsorbate_indices": adsorbate_indices,
        "per_atom_bader": [float(bader_charges[index]) for index in adsorbate_indices],
        "per_atom_reference": per_atom_reference,
        "assumed_elements": assumed,
    }


def combine_aeccar(
    aeccar0_path: str | Path,
    aeccar2_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Sum AECCAR0 and AECCAR2 into the reference density Bader prefers.

    This delegates to ``chgsum.pl`` from the Bader distribution, which has to be
    on ``PATH``.

    Args:
        aeccar0_path: Path to AECCAR0, the core charge density.
        aeccar2_path: Path to AECCAR2, the valence charge density.
        output_path: Path to write the combined density to.

    Returns:
        The path that was written.

    Raises:
        RuntimeError: If ``chgsum.pl`` is unavailable or did not produce a file.
    """
    aeccar0_path = Path(aeccar0_path)
    aeccar2_path = Path(aeccar2_path)
    output_path = Path(output_path)

    try:
        result = subprocess.run(
            ["chgsum.pl", aeccar0_path.name, aeccar2_path.name],
            cwd=aeccar0_path.parent,
            capture_output=True,
            check=False,
        )
    except OSError as exc:
        raise RuntimeError(f"Failed to combine AECCAR files: {exc}") from exc

    if result.returncode == 0:
        # chgsum.pl writes its output as CHGCAR_sum next to the inputs.
        chgcar_sum = aeccar0_path.parent / "CHGCAR_sum"
        if chgcar_sum.exists():
            import shutil

            shutil.move(str(chgcar_sum), str(output_path))
            return output_path

    raise RuntimeError(
        "Failed to combine AECCAR files: chgsum.pl produced no CHGCAR_sum. "
        "Make sure the Bader distribution's chgsum.pl is on PATH, or combine the files manually."
    )
