"""Relax a single structure with an MLP calculator.

Features:
  - DPA-3 (DeePMD-kit) and UMA (FAIRChem) machine-learned potentials;
  - selective dynamics inherited from the input file, or generated from a layer
    count or a fractional-z threshold;
  - fixed-cell or variable-cell relaxation;
  - a choice of optimizer (LBFGS, BFGS, FIRE, BFGSLineSearch) and convergence criteria.

Examples:
    perovml optimize input.vasp -o output.vasp
    perovml optimize input.vasp --calc uma --relax-cell
    perovml optimize input.vasp --calc dpa3 --fmax 0.01 --steps 500
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ase.atoms import Atoms

logger = logging.getLogger(__name__)

# ============================================================================
# Configuration (edit here to change the defaults)
# ============================================================================


class Config:
    """Default settings for :func:`optimize_structure`.

    Every command-line option falls back to the matching attribute here, so
    editing this class changes the defaults without touching the call sites.
    """

    # ------------------------------------------------------------------------
    # Calculator
    # ------------------------------------------------------------------------
    #: Calculator type: "dpa3" | "uma" | "mock".
    DEFAULT_CALCULATOR: str = "dpa3"

    #: Multi-task head of the DPA-3 model. The model path itself comes from
    #: --model-path, $DPA3_OMAT24_MODEL or $DPA3_MODEL_PATH.
    DPA3_HEAD: str = "Omat24"

    #: Pretrained UMA model name or local checkpoint path; $UMA_MODEL overrides it.
    UMA_MODEL: str = "uma-m-1p1"
    #: UMA task name, e.g. "oc20" or "omat".
    UMA_TASK: str = "omat"
    #: UMA inference mode: "default" or "turbo".
    UMA_INFERENCE: str = "default"
    #: UMA device: None picks one automatically, otherwise "cuda" or "cpu".
    UMA_DEVICE: str | None = None

    # ------------------------------------------------------------------------
    # Optimizer
    # ------------------------------------------------------------------------
    #: Optimizer name: "LBFGS" | "BFGS" | "FIRE" | "BFGSLineSearch".
    DEFAULT_OPTIMIZER: str = "LBFGS"

    #: Force convergence threshold, in eV/A.
    DEFAULT_FMAX: float = 0.01
    #: Maximum number of optimizer steps.
    DEFAULT_STEPS: int = 2000

    #: LBFGS: number of history vectors.
    LBFGS_MEMORY: int = 100
    #: LBFGS: damping factor.
    LBFGS_DAMPING: float = 1.0
    #: LBFGS: inverse of the initial Hessian diagonal.
    LBFGS_ALPHA: float = 70.0

    #: FIRE: maximum step length, in A.
    FIRE_MAXSTEP: float = 0.2
    #: FIRE: time step, in fs.
    FIRE_DT: float = 0.1

    # ------------------------------------------------------------------------
    # Cell relaxation
    # ------------------------------------------------------------------------
    #: Whether the cell is relaxed by default.
    DEFAULT_RELAX_CELL: bool = False

    #: Cell filter: "ExpCellFilter" | "UnitCellFilter" | "StrainFilter".
    CELL_FILTER: str = "ExpCellFilter"

    #: Cell "mass" factor; None lets ASE choose.
    CELL_FACTOR: float | None = None
    #: Allow hydrostatic strain only.
    HYDROSTATIC_STRAIN: bool = False
    #: Keep the volume constant.
    CONSTANT_VOLUME: bool = False
    #: External pressure, in GPa; 0 means none.
    SCALAR_PRESSURE: float = 0.0

    # ------------------------------------------------------------------------
    # Selective dynamics
    # ------------------------------------------------------------------------
    #: Keep the constraints already present in the input file.
    INHERIT_SELECTIVE_DYNAMICS: bool = True

    #: Fix the bottom N layers when the input carries no constraints; None disables it.
    AUTO_FIX_BOTTOM_LAYERS: int | None = None
    #: Or fix every atom below this fractional z; None disables it.
    AUTO_FIX_Z_THRESHOLD: float | None = None

    # ------------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------------
    #: ASE format the relaxed structure is written in.
    OUTPUT_FORMAT: str = "vasp"
    #: Write VASP output in fractional coordinates.
    VASP_DIRECT: bool = True
    #: Trajectory file name; None writes no trajectory.
    TRAJECTORY_FILE: str | None = None
    #: Log the energy and fmax every N steps.
    LOG_INTERVAL: int = 10


# ============================================================================
# Calculator, optimizer and constraint helpers
# ============================================================================


def resolve_dpa3_model_path(model_path: str | None = None) -> str:
    """Resolve the DPA-3 model path, preferring the explicit setting.

    Args:
        model_path: Path given on the command line, or None.

    Returns:
        The resolved model file path.

    Raises:
        ValueError: If no path was given and neither environment variable is set.
    """
    resolved = model_path or os.environ.get("DPA3_OMAT24_MODEL") or os.environ.get("DPA3_MODEL_PATH")
    if not resolved:
        raise ValueError(
            "The DPA-3 calculator needs a model path. Provide one of:\n"
            "  1. the --model-path command-line option\n"
            "  2. the DPA3_OMAT24_MODEL environment variable\n"
            "  3. the DPA3_MODEL_PATH environment variable\n"
            "See models/README.md for how to download the model."
        )
    return resolved


def get_calculator(calc_type: str, model_path: str | None = None):
    """Build the ASE calculator named by ``calc_type``.

    Args:
        calc_type: One of ``"dpa3"``, ``"uma"`` or ``"mock"``.
        model_path: Model path or pretrained model name, overriding the default.

    Returns:
        An ASE calculator.

    Raises:
        ImportError: If deepmd-kit or fairchem-core is missing.
        ValueError: If the calculator type is unknown or a model path is missing.
    """
    calc_type = calc_type.lower()

    if calc_type in ("dpa3", "dpa3_omat24", "dpa3omat24"):
        resolved = resolve_dpa3_model_path(model_path)

        # Import DeePMD-kit directly rather than through perovml.calculators, which
        # would pull in the heavy FAIRChem and torch stack (about five extra seconds).
        try:
            from deepmd.calculator import DP as DPCalculator
        except ImportError as exc:
            raise ImportError(
                "The DPA-3 calculator needs deepmd-kit. Install it with `pip install deepmd-kit`."
            ) from exc

        logger.info("loading DPA-3 model: %s", resolved)
        return DPCalculator(resolved, head=Config.DPA3_HEAD)

    if calc_type in ("uma", "fairchem", "fairchemcalculator"):
        try:
            from fairchem.core import FAIRChemCalculator
        except ImportError as exc:
            raise ImportError(
                "The UMA calculator needs fairchem. Install it with `pip install fairchem-core`."
            ) from exc

        uma_model = model_path or os.environ.get("UMA_MODEL") or Config.UMA_MODEL
        if not uma_model:
            raise ValueError(
                "The UMA calculator needs a model name or path. Provide one of:\n"
                "  1. the --model-path command-line option\n"
                "  2. the UMA_MODEL environment variable"
            )

        # Accept "uma-m-1p1.pt" as a way of naming the pretrained model "uma-m-1p1".
        name_or_path = uma_model
        if name_or_path.endswith(".pt") and not os.path.isfile(name_or_path):
            name_or_path = os.path.basename(name_or_path)[:-3]

        logger.info("loading UMA model: %s, task=%s", name_or_path, Config.UMA_TASK)
        return FAIRChemCalculator.from_model_checkpoint(
            name_or_path=name_or_path,
            task_name=Config.UMA_TASK,
            inference_settings=Config.UMA_INFERENCE,
            device=Config.UMA_DEVICE,
        )

    if calc_type == "mock":
        from perovml.calculators.mock import MockCalculator

        logger.info("using the mock calculator (testing only)")
        return MockCalculator()

    raise ValueError(f"Unknown calculator type: {calc_type}")


def get_optimizer(opt_name: str):
    """Look up an ASE optimizer class by name, ignoring case.

    Args:
        opt_name: Optimizer name.

    Returns:
        The ASE optimizer class.

    Raises:
        ValueError: If the optimizer name is unknown.
    """
    from ase.optimize import LBFGS

    opt_map = {"LBFGS": LBFGS}

    try:
        from ase.optimize import BFGS, FIRE

        opt_map["BFGS"] = BFGS
        opt_map["FIRE"] = FIRE
    except ImportError:
        pass

    try:
        from ase.optimize import BFGSLineSearch

        opt_map["BFGSLineSearch"] = BFGSLineSearch
    except ImportError:
        pass

    opt_name = opt_name.upper()
    for key in opt_map:
        if key.upper() == opt_name:
            return opt_map[key]
    raise ValueError(f"Unknown optimizer: {opt_name}. Available: {list(opt_map.keys())}")


def get_cell_filter(filter_name: str):
    """Look up an ASE cell filter class by name.

    Args:
        filter_name: Cell filter name.

    Returns:
        The ASE cell filter class.

    Raises:
        ValueError: If the filter name is unknown or unavailable in this ASE version.
    """
    try:
        from ase.constraints import ExpCellFilter, StrainFilter, UnitCellFilter

        filter_map = {
            "ExpCellFilter": ExpCellFilter,
            "UnitCellFilter": UnitCellFilter,
            "StrainFilter": StrainFilter,
        }
    except ImportError:
        # Older ASE releases ship a smaller set.
        from ase.constraints import UnitCellFilter

        filter_map = {"UnitCellFilter": UnitCellFilter}
        try:
            from ase.constraints import ExpCellFilter

            filter_map["ExpCellFilter"] = ExpCellFilter
        except ImportError:
            pass

    if filter_name not in filter_map:
        raise ValueError(f"Unknown cell filter: {filter_name}. Available: {list(filter_map.keys())}")

    return filter_map[filter_name]


def apply_selective_dynamics(atoms: Atoms, inherit: bool = True) -> Atoms:
    """Attach the selective-dynamics constraints the relaxation should honour.

    Constraints already carried by the input file win. Failing that, the
    ``AUTO_FIX_*`` settings on :class:`Config` can generate them; if neither
    applies, every atom stays free.

    Args:
        atoms: Input structure. Modified in place.
        inherit: Whether to keep constraints already present in the input file.

    Returns:
        The constrained structure.
    """
    from ase.constraints import FixAtoms, FixCartesian

    # ASE turns a POSCAR's selective-dynamics block into FixScaled/FixCartesian
    # constraints on read, so look for those first.
    existing_constraints = atoms.constraints
    if inherit and existing_constraints:
        fix_atoms_count = 0
        partial_fix_count = 0
        for constraint in existing_constraints:
            name = type(constraint).__name__
            if name == "FixAtoms":
                fix_atoms_count += len(constraint.index)
            elif name in ("FixScaled", "FixCartesian"):
                partial_fix_count += 1

        if fix_atoms_count > 0:
            logger.info("inherited constraints: %d atoms fully fixed", fix_atoms_count)
        if partial_fix_count > 0:
            logger.info("inherited constraints: %d atoms fixed along some directions", partial_fix_count)
        if fix_atoms_count == 0 and partial_fix_count == 0:
            logger.info("inherited constraints: %d constraint objects", len(existing_constraints))
        return atoms

    # Some ASE versions expose the raw flags as an array instead.
    if inherit and hasattr(atoms, "arrays") and "selective_dynamics" in atoms.arrays:
        selective_dynamics = atoms.arrays["selective_dynamics"]

        fix_indices: list[int] = []
        partial_fix: list[tuple[int, list[bool]]] = []

        for index, flags in enumerate(selective_dynamics):
            if not any(flags):  # all F: fully fixed
                fix_indices.append(index)
            elif not all(flags):  # partially fixed
                partial_fix.append((index, flags))

        constraints: list[object] = []

        if fix_indices:
            constraints.append(FixAtoms(indices=fix_indices))
            logger.info("inherited selective dynamics: %d atoms fixed", len(fix_indices))

        # VASP writes T for a free direction, ASE's FixCartesian mask marks the
        # fixed ones, so the flags have to be inverted.
        for index, flags in partial_fix:
            constraints.append(FixCartesian(index, mask=[not flag for flag in flags]))

        if partial_fix:
            logger.info("partial constraints: %d atoms fixed along some directions", len(partial_fix))

        if constraints:
            atoms.set_constraint(constraints)
        return atoms

    # No selective dynamics in the input: fall back to the automatic rules.
    if Config.AUTO_FIX_BOTTOM_LAYERS is not None:
        atoms = fix_bottom_layers(atoms, Config.AUTO_FIX_BOTTOM_LAYERS)
    elif Config.AUTO_FIX_Z_THRESHOLD is not None:
        atoms = fix_by_z_threshold(atoms, Config.AUTO_FIX_Z_THRESHOLD)
    else:
        logger.info("no constraints, every atom relaxes freely")

    return atoms


def fix_bottom_layers(atoms: Atoms, n_layers: int) -> Atoms:
    """Fix the bottom ``n_layers`` layers, grouped by fractional z.

    Args:
        atoms: Input structure. Modified in place.
        n_layers: Number of bottom layers to fix.

    Returns:
        The constrained structure, unchanged when it has fewer than ``n_layers`` layers.
    """
    import numpy as np
    from ase.constraints import FixAtoms

    z_coords = atoms.get_scaled_positions()[:, 2]
    unique_z = np.unique(np.round(z_coords, decimals=3))

    if len(unique_z) <= n_layers:
        logger.warning("structure has only %d layers, cannot fix %d", len(unique_z), n_layers)
        return atoms

    threshold_z = unique_z[n_layers - 1] + 0.01  # small tolerance
    fix_indices = np.where(z_coords <= threshold_z)[0].tolist()

    if fix_indices:
        atoms.set_constraint(FixAtoms(indices=fix_indices))
        logger.info("fixed the bottom %d layers (%d atoms)", n_layers, len(fix_indices))

    return atoms


def fix_by_z_threshold(atoms: Atoms, z_threshold: float) -> Atoms:
    """Fix every atom below a fractional-z threshold.

    Args:
        atoms: Input structure. Modified in place.
        z_threshold: Fractional-z threshold; atoms below it are fixed.

    Returns:
        The constrained structure.
    """
    import numpy as np
    from ase.constraints import FixAtoms

    z_coords = atoms.get_scaled_positions()[:, 2]
    fix_indices = np.where(z_coords < z_threshold)[0].tolist()

    if fix_indices:
        atoms.set_constraint(FixAtoms(indices=fix_indices))
        logger.info("fixed the %d atoms below z = %.3f", len(fix_indices), z_threshold)

    return atoms


# ============================================================================
# Relaxation
# ============================================================================


def optimize_structure(
    input_file: str,
    output_file: str | None = None,
    calc_type: str | None = None,
    optimizer: str | None = None,
    fmax: float | None = None,
    steps: int | None = None,
    relax_cell: bool | None = None,
    model_path: str | None = None,
    trajectory: str | None = None,
) -> Atoms:
    """Relax one structure and write it out.

    Every argument left as None falls back to the matching :class:`Config` attribute.

    Args:
        input_file: Path to the input structure, in any ASE-readable format.
        output_file: Path to write the relaxed structure to. Defaults to the input
            name with an ``_opt`` suffix.
        calc_type: Calculator type, see :func:`get_calculator`.
        optimizer: Optimizer name, see :func:`get_optimizer`.
        fmax: Force convergence threshold, in eV/A.
        steps: Maximum number of optimizer steps.
        relax_cell: Whether to relax the cell as well as the ions.
        model_path: Model path or pretrained model name, overriding the default.
        trajectory: Path for the optimizer trajectory, or None to write none.

    Returns:
        The relaxed structure, with its calculator detached.
    """
    import time

    from ase.io import read, write

    calc_type = calc_type or Config.DEFAULT_CALCULATOR
    optimizer = optimizer or Config.DEFAULT_OPTIMIZER
    fmax = fmax if fmax is not None else Config.DEFAULT_FMAX
    steps = steps if steps is not None else Config.DEFAULT_STEPS
    relax_cell = relax_cell if relax_cell is not None else Config.DEFAULT_RELAX_CELL
    trajectory = trajectory or Config.TRAJECTORY_FILE

    if output_file is None:
        input_path = Path(input_file)
        output_file = str(input_path.parent / f"{input_path.stem}_opt{input_path.suffix}")

    logger.info("perovml optimize: %s -> %s", input_file, output_file)
    logger.info(
        "calculator=%s optimizer=%s fmax=%s eV/A steps=%s relax_cell=%s",
        calc_type,
        optimizer,
        fmax,
        steps,
        relax_cell,
    )

    # 1. Read the structure.
    atoms = read(input_file)
    logger.info("input: %s, %d atoms, cell %s", atoms.get_chemical_formula(), len(atoms), atoms.cell.lengths())

    # 2. Apply selective dynamics.
    atoms = apply_selective_dynamics(atoms, inherit=Config.INHERIT_SELECTIVE_DYNAMICS)

    # 3. Attach the calculator.
    calc = get_calculator(calc_type, model_path=model_path)
    atoms.calc = calc

    # 4. Initial energy and forces.
    e_init = atoms.get_potential_energy()
    fmax_init = abs(atoms.get_forces()).max()
    logger.info("initial energy %.6f eV, initial fmax %.6f eV/A", e_init, fmax_init)

    # 5. Decide what the optimizer acts on.
    opt_target: object = atoms

    if relax_cell:
        cell_filter_cls = get_cell_filter(Config.CELL_FILTER)
        filter_kwargs: dict[str, object] = {}

        if Config.CELL_FACTOR is not None:
            filter_kwargs["cell_factor"] = Config.CELL_FACTOR
        if Config.HYDROSTATIC_STRAIN:
            filter_kwargs["hydrostatic_strain"] = True
        if Config.CONSTANT_VOLUME:
            filter_kwargs["constant_volume"] = True
        if Config.SCALAR_PRESSURE != 0.0:
            filter_kwargs["scalar_pressure"] = Config.SCALAR_PRESSURE

        opt_target = cell_filter_cls(atoms, **filter_kwargs)
        logger.info("relaxing the cell with %s", Config.CELL_FILTER)

    # 6. Build the optimizer.
    optimizer_cls = get_optimizer(optimizer)
    opt_kwargs: dict[str, object] = {}

    if trajectory:
        opt_kwargs["trajectory"] = trajectory
    if optimizer.upper() == "LBFGS":
        opt_kwargs["memory"] = Config.LBFGS_MEMORY
        opt_kwargs["damping"] = Config.LBFGS_DAMPING
        opt_kwargs["alpha"] = Config.LBFGS_ALPHA
    elif optimizer.upper() == "FIRE":
        opt_kwargs["maxstep"] = Config.FIRE_MAXSTEP
        opt_kwargs["dt"] = Config.FIRE_DT

    dyn = optimizer_cls(opt_target, **opt_kwargs)

    # 7. Relax.
    start_time = time.time()
    step_count = [0]

    def log_step() -> None:
        """Report the energy and fmax every ``Config.LOG_INTERVAL`` steps."""
        step_count[0] += 1
        if step_count[0] % Config.LOG_INTERVAL == 0 or step_count[0] == 1:
            energy = atoms.get_potential_energy()
            current_fmax = abs(atoms.get_forces()).max()
            logger.info("step %4d: E = %12.6f eV, fmax = %.6f eV/A", step_count[0], energy, current_fmax)

    dyn.attach(log_step)

    converged = dyn.run(fmax=fmax, steps=steps)
    elapsed = time.time() - start_time

    # 8. Summary.
    e_final = atoms.get_potential_energy()
    fmax_final = abs(atoms.get_forces()).max()

    logger.info("%s after %d steps in %.2f s", "converged" if converged else "not converged", step_count[0], elapsed)
    logger.info("energy %.6f -> %.6f eV (delta = %.6f eV)", e_init, e_final, e_final - e_init)
    logger.info("fmax %.6f -> %.6f eV/A", fmax_init, fmax_final)
    if relax_cell:
        logger.info("final cell: %s", atoms.cell.lengths())

    # 9. Write the result. The calculator is detached first so that the structure
    # stays serialisable.
    atoms.calc = None
    write(output_file, atoms, format=Config.OUTPUT_FORMAT, direct=Config.VASP_DIRECT)
    logger.info("wrote %s", output_file)
    if trajectory:
        logger.info("wrote trajectory %s", trajectory)

    return atoms


# ============================================================================
# Command-line front end
# ============================================================================


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``perovml optimize`` arguments.

    Args:
        parser: Subcommand parser to populate.
    """
    parser.epilog = """
Examples:
  # relax with DPA-3
  perovml optimize input.vasp -o output.vasp --calc dpa3

  # relax with UMA, cell included
  perovml optimize input.vasp --calc uma --relax-cell

  # tighter convergence
  perovml optimize input.vasp --fmax 0.01 --steps 500

  # use the FIRE optimizer
  perovml optimize input.vasp --optimizer FIRE

  # name the model file explicitly (or export DPA3_MODEL_PATH=...)
  perovml optimize input.vasp --calc dpa3 --model-path DPA-3.1-3M.pt
"""

    parser.add_argument("input", help="Input structure file (VASP, CIF, XYZ, ...)")
    parser.add_argument("-o", "--output", help="Output file name (default: <input>_opt.vasp)")

    parser.add_argument(
        "--calc",
        "--calculator",
        dest="calc",
        choices=["dpa3", "uma", "mock"],
        default=None,
        help=f"Calculator type (default: {Config.DEFAULT_CALCULATOR})",
    )
    parser.add_argument(
        "--model-path",
        help="ML model path or pretrained model name (or set DPA3_MODEL_PATH / UMA_MODEL)",
    )

    parser.add_argument(
        "--optimizer",
        "-O",
        choices=["LBFGS", "BFGS", "FIRE", "BFGSLineSearch"],
        default=None,
        help=f"Optimization algorithm (default: {Config.DEFAULT_OPTIMIZER})",
    )
    parser.add_argument(
        "--fmax",
        type=float,
        default=None,
        help=f"Force convergence threshold in eV/A (default: {Config.DEFAULT_FMAX})",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=None,
        help=f"Maximum optimizer steps (default: {Config.DEFAULT_STEPS})",
    )

    cell_grp = parser.add_mutually_exclusive_group()
    cell_grp.add_argument("--relax-cell", action="store_true", help="Relax the cell as well as the ions")
    cell_grp.add_argument("--fix-cell", action="store_true", help="Keep the cell fixed (default)")

    parser.add_argument("--trajectory", "-t", help="Write the optimizer trajectory to this .traj file")


def run(args: argparse.Namespace) -> int:
    """Execute ``perovml optimize``.

    Args:
        args: Parsed arguments from :func:`add_arguments`.

    Returns:
        Process exit status.
    """
    relax_cell = None
    if args.relax_cell:
        relax_cell = True
    elif args.fix_cell:
        relax_cell = False

    optimize_structure(
        input_file=args.input,
        output_file=args.output,
        calc_type=args.calc,
        optimizer=args.optimizer,
        fmax=args.fmax,
        steps=args.steps,
        relax_cell=relax_cell,
        model_path=args.model_path,
        trajectory=args.trajectory,
    )
    return 0
