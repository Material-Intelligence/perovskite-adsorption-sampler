"""Place a probe molecule on a Pb-I terminated slab and rank the configurations.

The script exercises both samplers shipped in :class:`perovml.PbAdsorbateSlabConfig`
on one slab and one probe molecule, relaxes every configuration with
:class:`perovml.MockCalculator`, prints a ranked table and writes the best
structure as a POSCAR.

The mock calculator is a quadratic well: it needs no model file, no network and
no GPU, and the energies it returns are meaningless. The point of this example is
to show the call sequence -- read structures, tag the surface, sample
orientations, relax, rank, write -- not to produce physical numbers. Swap
``MockCalculator`` for :class:`perovml.calculators.dpa3.DPA3Omat24Calculator` (or
any other ASE calculator) to get meaningful energies.

Runtime: 4-5 s wall clock with the default settings on an Apple M-series CPU
(Python 3.11, 20 configurations). Roughly 4 s of that is the FAIRChem and torch
import; the sampling, relaxation and ranking take 0.3 s.

Usage:
    python examples/01_place_and_screen.py
    python examples/01_place_and_screen.py --num-orientations 20 --output-dir example_output
"""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path

from ase.atoms import Atoms
from ase.optimize import LBFGS
from monty.json import MSONable
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.io.vasp import Poscar

from perovml import MockCalculator, PbAdsorbateSlabConfig
from perovml.utils.structure import build_adsorbate, build_oc_slab_from_atoms

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SLAB = REPO_ROOT / "data" / "slabs" / "FAPbI3_3x3_zcut017.vasp"
DEFAULT_MOLECULE = REPO_ROOT / "data" / "molecules" / "Acetone.vasp"

SAMPLERS = ("Pb_uniform_sample", "Pb_heuristic_sample")


@dataclass
class ScreeningResult(MSONable):
    """One relaxed adsorbate-slab configuration and its mock energies.

    Attributes:
        name: Configuration name, e.g. ``"uniform_003"``.
        sampler: Sampler that produced the configuration, one of
            ``"Pb_uniform_sample"`` or ``"Pb_heuristic_sample"``.
        index: Index of the configuration within its sampler.
        energy: Total energy of the relaxed adsorbate-slab system, in eV.
        adsorption_energy: ``energy`` minus the relaxed slab and relaxed gas-phase
            molecule energies, in eV.
    """

    name: str
    sampler: str
    index: int
    energy: float
    adsorption_energy: float

    def as_dict(self) -> dict:
        """Serialize to an MSON-compatible dictionary.

        Returns:
            A dictionary carrying ``@module`` / ``@class`` keys plus every field.
        """
        return {
            "@module": type(self).__module__,
            "@class": type(self).__name__,
            "name": self.name,
            "sampler": self.sampler,
            "index": self.index,
            "energy": self.energy,
            "adsorption_energy": self.adsorption_energy,
        }

    @classmethod
    def from_dict(cls, dct: dict) -> ScreeningResult:
        """Rebuild a result from :meth:`as_dict` output.

        Args:
            dct: Dictionary produced by :meth:`as_dict`.

        Returns:
            The reconstructed :class:`ScreeningResult`.
        """
        return cls(
            name=dct["name"],
            sampler=dct["sampler"],
            index=int(dct["index"]),
            energy=float(dct["energy"]),
            adsorption_energy=float(dct["adsorption_energy"]),
        )


def read_structure(path: str | Path) -> Atoms:
    """Read a structure file with pymatgen and hand it to ASE.

    Args:
        path: Path to any file :meth:`pymatgen.core.Structure.from_file` accepts,
            for instance a POSCAR or a CIF.

    Returns:
        The structure as an ASE :class:`~ase.atoms.Atoms` object.
    """
    structure = Structure.from_file(str(path))
    return AseAtomsAdaptor.get_atoms(structure)


def relax(atoms: Atoms, fmax: float, steps: int) -> tuple[Atoms, float]:
    """Relax a structure with the mock calculator.

    Args:
        atoms: Structure to relax. A copy is relaxed; the input is left alone.
        fmax: Force convergence threshold, in eV/A.
        steps: Maximum number of optimizer steps.

    Returns:
        A ``(relaxed_atoms, energy)`` tuple, with the energy in eV.
    """
    working = atoms.copy()
    working.calc = MockCalculator()
    LBFGS(working, logfile=None).run(fmax=fmax, steps=steps)
    return working, float(working.get_potential_energy())


def screen(
    slab_file: Path,
    molecule_file: Path,
    num_orientations: int,
    cone_angle_deg: float,
    fmax: float,
    steps: int,
    min_ab: float,
    seed: int,
) -> tuple[list[ScreeningResult], dict[str, Atoms]]:
    """Sample, relax and rank adsorbate-slab configurations.

    Both samplers are run on the same slab and molecule: ``Pb_uniform_sample``
    draws uniform SO(3) orientations anchored at the molecule's centre of mass,
    ``Pb_heuristic_sample`` draws orientations inside a cone around the surface
    normal, anchored at the binding atom.

    Args:
        slab_file: Slab structure file, Pb-I terminated.
        molecule_file: Probe molecule structure file.
        num_orientations: Orientations to generate per sampler.
        cone_angle_deg: Cone half-angle for ``Pb_heuristic_sample``, in degrees.
        fmax: Force convergence threshold, in eV/A.
        steps: Maximum number of optimizer steps per configuration.
        min_ab: Minimum in-plane lattice vector length the FAIRChem ``Slab``
            constructor requires, in Angstrom.
        seed: Seed for ``Pb_uniform_sample``. The same seed gives the same
            candidate set; ``Pb_heuristic_sample`` is deterministic and ignores it.

    Returns:
        A ``(results, structures)`` tuple. ``results`` is sorted by adsorption
        energy, lowest first; ``structures`` maps a configuration name to its
        relaxed structure.
    """
    slab_atoms = read_structure(slab_file)
    slab = build_oc_slab_from_atoms(slab_atoms, min_ab=min_ab)
    adsorbate = build_adsorbate(molecule_file=str(molecule_file))

    # Reference energies, so that the table can report an adsorption energy
    # rather than a total energy dominated by the slab.
    _, slab_energy = relax(slab.atoms, fmax=fmax, steps=steps)
    _, molecule_energy = relax(adsorbate.atoms, fmax=fmax, steps=steps)

    results: list[ScreeningResult] = []
    structures: dict[str, Atoms] = {}

    for sampler in SAMPLERS:
        config = PbAdsorbateSlabConfig(
            slab=slab,
            adsorbate=adsorbate,
            num_orientations=num_orientations,
            cone_angle_deg=cone_angle_deg,
            mode=sampler,
            rng=seed,
        )
        prefix = "uniform" if sampler == "Pb_uniform_sample" else "heuristic"
        for index, candidate in enumerate(config.atoms_list):
            relaxed, energy = relax(candidate, fmax=fmax, steps=steps)
            name = f"{prefix}_{index:03d}"
            structures[name] = relaxed
            results.append(
                ScreeningResult(
                    name=name,
                    sampler=sampler,
                    index=index,
                    energy=energy,
                    adsorption_energy=energy - slab_energy - molecule_energy,
                )
            )

    results.sort(key=lambda result: result.adsorption_energy)
    return results, structures


def print_table(results: list[ScreeningResult], top: int) -> None:
    """Print the ranked configurations.

    Args:
        results: Results sorted by adsorption energy, lowest first.
        top: Number of rows to print.
    """
    best = results[0].adsorption_energy
    print(f"{'rank':>4}  {'config':<14}  {'sampler':<20}  {'E_ads (eV)':>12}  {'dE (eV)':>10}")
    print("-" * 68)
    for rank, result in enumerate(results[:top], start=1):
        delta = result.adsorption_energy - best
        print(
            f"{rank:>4}  {result.name:<14}  {result.sampler:<20}  " f"{result.adsorption_energy:>12.4f}  {delta:>10.4f}"
        )
    if len(results) > top:
        print(f"... {len(results) - top} more configurations not shown")


def write_best(result: ScreeningResult, atoms: Atoms, output_dir: Path) -> Path:
    """Write the best configuration as a POSCAR.

    Species are grouped, as VASP and most viewers expect, and the selective
    dynamics flags carried over from the slab constraints are kept.

    Args:
        result: The top-ranked result.
        atoms: The relaxed structure belonging to ``result``.
        output_dir: Directory to write into. Created if missing.

    Returns:
        Path of the file written.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    structure = AseAtomsAdaptor.get_structure(atoms)
    poscar = Poscar(
        structure,
        comment=f"{result.name} ({result.sampler}), mock E_ads = {result.adsorption_energy:.4f} eV",
        sort_structure=True,
    )
    path = output_dir / "best_adslab.vasp"
    poscar.write_file(str(path))
    return path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--slab", type=Path, default=DEFAULT_SLAB, help="slab structure file")
    parser.add_argument("--molecule", type=Path, default=DEFAULT_MOLECULE, help="probe molecule structure file")
    parser.add_argument(
        "--num-orientations", type=int, default=10, help="orientations per sampler (two samplers are run)"
    )
    parser.add_argument("--cone-angle", type=float, default=25.0, help="cone half-angle for Pb_heuristic_sample, deg")
    parser.add_argument("--fmax", type=float, default=0.05, help="force convergence threshold, eV/A")
    parser.add_argument("--steps", type=int, default=30, help="maximum optimizer steps per configuration")
    parser.add_argument("--min-ab", type=float, default=8.0, help="minimum in-plane lattice vector length, A")
    parser.add_argument("--seed", type=int, default=0, help="seed for the uniform sampler")
    parser.add_argument("--top", type=int, default=10, help="rows to print in the ranked table")
    parser.add_argument("--output-dir", type=Path, default=Path("example_output"), help="directory for the best POSCAR")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    """Run the example end to end.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.
    """
    args = parse_args(argv)
    started = time.perf_counter()

    slab_formula = Structure.from_file(str(args.slab)).composition.reduced_formula
    molecule_formula = Structure.from_file(str(args.molecule)).composition.reduced_formula
    print(f"slab     : {args.slab.name} ({slab_formula})")
    print(f"molecule : {args.molecule.name} ({molecule_formula})")
    print(f"sampling : {args.num_orientations} orientations x {len(SAMPLERS)} samplers, mock calculator\n")

    results, structures = screen(
        slab_file=args.slab,
        molecule_file=args.molecule,
        num_orientations=args.num_orientations,
        cone_angle_deg=args.cone_angle,
        fmax=args.fmax,
        steps=args.steps,
        min_ab=args.min_ab,
        seed=args.seed,
    )

    print()
    print_table(results, top=args.top)

    best = results[0]
    path = write_best(best, structures[best.name], args.output_dir)
    print(f"\nbest configuration written to {path}")
    print(f"total wall clock: {time.perf_counter() - started:.1f} s")
    print("\nThe energies above come from the mock calculator and carry no physical meaning.")


if __name__ == "__main__":
    main()
