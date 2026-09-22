"""Write VASP selective-dynamics flags onto a POSCAR.

Two rules are available. ``--z-cut`` relaxes every site above a fractional-z
threshold and freezes the rest, which is the usual slab setup. ``--uniform``
applies the same flags to every site, e.g. ``T T F`` to allow in-plane motion only.

Examples:
    perovml sd data/slabs/FAPbI3_3x3_zcut017.vasp -o slab_sd.vasp --z-cut 0.26
    perovml sd data/slabs/FAPbI3_3x3.vasp -o slab_xy.vasp --uniform TTF
"""

from __future__ import annotations

import argparse


def _parse_flags(text: str) -> tuple[bool, bool, bool]:
    """Parse a three-character selective-dynamics flag string.

    Args:
        text: Three ``T``/``F`` characters, optionally separated by spaces
            (e.g. ``"TTF"`` or ``"T T F"``).

    Returns:
        The ``(x, y, z)`` flags, where True means the direction is relaxed.

    Raises:
        argparse.ArgumentTypeError: If the string is not three T/F characters.
    """
    cleaned = text.replace(" ", "").upper()
    if len(cleaned) != 3 or set(cleaned) - {"T", "F"}:
        raise argparse.ArgumentTypeError(f"expected three T/F characters, e.g. TTF, got {text!r}")
    return tuple(char == "T" for char in cleaned)  # type: ignore[return-value]


def add_arguments(parser: argparse.ArgumentParser) -> None:
    """Register the ``perovml sd`` arguments.

    Args:
        parser: Subcommand parser to populate.
    """
    parser.add_argument("input", help="Input POSCAR/CONTCAR path")
    parser.add_argument(
        "-o",
        "--output",
        help="Output POSCAR path (default: <input stem>_sd<suffix> next to the input)",
    )

    rule = parser.add_mutually_exclusive_group(required=True)
    rule.add_argument(
        "--z-cut",
        type=float,
        help="Fractional z threshold: sites above it are relaxed, the rest are frozen",
    )
    rule.add_argument(
        "--uniform",
        type=_parse_flags,
        metavar="FLAGS",
        help="Apply the same three T/F flags to every site, e.g. TTF",
    )

    parser.add_argument(
        "--free-flags",
        type=_parse_flags,
        default="TTT",
        metavar="FLAGS",
        help="Flags for sites above --z-cut (default: TTT)",
    )
    parser.add_argument(
        "--fixed-flags",
        type=_parse_flags,
        default="FFF",
        metavar="FLAGS",
        help="Flags for sites at or below --z-cut (default: FFF)",
    )


def run(args: argparse.Namespace) -> int:
    """Read the input POSCAR, apply the selected rule and write the result.

    Args:
        args: Parsed arguments from :func:`add_arguments`.

    Returns:
        Process exit status.
    """
    from pathlib import Path

    try:
        from pymatgen.io.vasp.inputs import Poscar
    except ImportError as exc:  # pragma: no cover - runtime import guard
        raise ImportError("`perovml sd` requires pymatgen. Install it with `pip install pymatgen`.") from exc

    from perovml.utils.poscar_tools import selective_dynamics_by_z, uniform_selective_dynamics

    in_path = Path(args.input)
    poscar = Poscar.from_file(str(in_path))

    if args.uniform is not None:
        out = uniform_selective_dynamics(poscar.structure, flags=args.uniform, comment=poscar.comment)
    else:
        out = selective_dynamics_by_z(
            poscar.structure,
            z_cut=args.z_cut,
            free_flags=args.free_flags,
            fixed_flags=args.fixed_flags,
            comment=poscar.comment,
        )

    out_path = Path(args.output) if args.output else in_path.with_name(f"{in_path.stem}_sd{in_path.suffix}")
    out.write_file(str(out_path))

    n_free = sum(1 for flags in out.selective_dynamics if any(flags))
    print(f"Wrote {out_path}")
    print(f"  sites: {len(out.structure)}  fully or partly relaxed: {n_free}")
    return 0
