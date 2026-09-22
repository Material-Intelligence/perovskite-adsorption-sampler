#!/usr/bin/env python3
"""Thin wrapper around ``perovml place``.

Kept for people who prefer running a script out of a checkout to installing the
console script. It forwards every argument unchanged:

    python scripts/gen_from_slab.py --slab-file data/slabs/FAPbI3_3x3_zcut017.vasp \\
        --molecule-file data/molecules/Acetone.vasp --out-root outputs/placements

is the same as::

    perovml place --slab-file ... --molecule-file ... --out-root ...
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from perovml.cli.main import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main(["place", *sys.argv[1:]]))
