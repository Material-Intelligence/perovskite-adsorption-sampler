"""Entry point for the ``perovml`` console script.

The command is a thin argparse dispatcher over the subcommand modules in
:mod:`perovml.cli`. Each of those modules exposes ``add_arguments(parser)`` and
``run(args)``, and keeps its heavy imports (FAIRChem, DeePMD-kit, torch) inside
``run`` so that ``perovml --help`` works on a bare install.

Subcommands:
    run: Sample adsorbate placements and relax them, driven by a YAML config.
    place: Generate adsorbate placements only, without any energy evaluation.
    optimize: Relax a single structure with an MLP calculator.
    sd: Write VASP selective-dynamics flags onto a POSCAR.
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence

from perovml import __version__
from perovml.cli import optimize, place, run, sd

_SUBCOMMANDS = (
    ("run", run, "Sample and relax adsorbate placements from a YAML config"),
    ("place", place, "Generate adsorbate placements without relaxing them"),
    ("optimize", optimize, "Relax one structure with an MLP calculator"),
    ("sd", sd, "Set VASP selective-dynamics flags on a POSCAR"),
)


def build_parser() -> argparse.ArgumentParser:
    """Build the top-level argument parser with every subcommand attached.

    Returns:
        The configured parser.
    """
    parser = argparse.ArgumentParser(
        prog="perovml",
        description="Adsorbate placement sampling and MLP-to-DFT workflows for perovskite surfaces.",
    )
    parser.add_argument("--version", action="version", version=f"perovml {__version__}")

    subparsers = parser.add_subparsers(dest="command", title="subcommands")
    for name, module, help_text in _SUBCOMMANDS:
        subparser = subparsers.add_parser(
            name,
            help=help_text,
            description=module.__doc__,
            formatter_class=argparse.RawDescriptionHelpFormatter,
        )
        module.add_arguments(subparser)
        subparser.set_defaults(func=module.run)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments and dispatch to the selected subcommand.

    Args:
        argv: Argument list. Defaults to ``sys.argv[1:]``.

    Returns:
        The subcommand's exit status.
    """
    # The package logs progress through the standard logging module and installs no
    # handler of its own; the console script is the place that decides where it goes.
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

    parser = build_parser()
    args = parser.parse_args(argv)

    if getattr(args, "func", None) is None:
        parser.print_help()
        return 1

    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
