#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _save_json_atomic(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    tmp.replace(path)


def _iter_adslab_dirs(output_root: Path) -> list[Path]:
    molecules_dir = output_root / "molecules"
    if not molecules_dir.exists():
        return []
    out: list[Path] = []
    for mol_dir in sorted(molecules_dir.iterdir()):
        if not mol_dir.is_dir():
            continue
        status = _load_json(mol_dir / "status.json")
        if not status.get("mlp_done", False):
            continue
        adslab = mol_dir / "dft" / "adslab"
        if adslab.is_dir():
            out.append(adslab)
    return out


_INCAR_KEY_RE = re.compile(r"^\s*([A-Za-z0-9_]+)\s*=")


def _update_incar_lines(lines: list[str], updates: dict[str, str]) -> list[str]:
    # VASP comments often start with "!" (also allow "#").
    remaining = {k.upper(): v for k, v in updates.items()}
    out: list[str] = []
    for line in lines:
        m = _INCAR_KEY_RE.match(line)
        if not m:
            out.append(line)
            continue
        key = m.group(1).upper()
        if key not in remaining:
            out.append(line)
            continue
        # Preserve trailing comments
        comment = ""
        for sep in ("!", "#"):
            if sep in line:
                # keep the first comment marker
                parts = line.split(sep, 1)
                line = parts[0]
                comment = sep + parts[1]
                break
        new_val = remaining.pop(key)
        out.append(f"{key} = {new_val}{(' ' if comment and not comment.startswith(' ') else '')}{comment}".rstrip())
    # append missing keys at the end
    for key, val in remaining.items():
        out.append(f"{key} = {val}")
    # ensure newline termination by caller
    return out


def update_incar(path: Path, *, ncore: int, ediffg: float) -> None:
    text = path.read_text().splitlines()
    updates = {
        "NCORE": str(int(ncore)),
        "EDIFFG": str(float(ediffg)),
    }
    new_lines = _update_incar_lines(text, updates)
    path.write_text("\n".join(new_lines) + "\n")


@dataclass(frozen=True)
class PrepareResult:
    adslab_dir: Path
    prepared: bool
    reason: str


def prepare_one(
    adslab_dir: Path,
    *,
    ncore: int,
    ediffg: float,
    subvasp_src: Path,
    state: dict[str, Any],
    state_path: Path,
) -> PrepareResult:
    key = str(adslab_dir.resolve())
    prepared = state.setdefault("prepared", {})
    if key in prepared:
        return PrepareResult(adslab_dir=adslab_dir, prepared=False, reason="already_prepared")

    incar = adslab_dir / "INCAR"
    if not incar.exists():
        return PrepareResult(adslab_dir=adslab_dir, prepared=False, reason="missing_INCAR")

    if not subvasp_src.exists():
        raise FileNotFoundError(f"subvasp.sh not found: {subvasp_src}")

    update_incar(incar, ncore=ncore, ediffg=ediffg)
    shutil.copy2(subvasp_src, adslab_dir / "subvasp.sh")

    prepared[key] = {"at": _now(), "ncore": int(ncore), "ediffg": float(ediffg)}
    _save_json_atomic(state_path, state)
    return PrepareResult(adslab_dir=adslab_dir, prepared=True, reason="prepared")


def submit_one(
    adslab_dir: Path,
    *,
    state: dict[str, Any],
    state_path: Path,
) -> tuple[bool, str]:
    key = str(adslab_dir.resolve())
    prepared = state.setdefault("prepared", {})
    submitted = state.setdefault("submitted", {})

    if key in submitted:
        return False, "already_submitted"
    if key not in prepared:
        return False, "not_prepared"
    script = adslab_dir / "subvasp.sh"
    if not script.exists():
        return False, "missing_subvasp.sh"

    proc = subprocess.run(
        ["sbatch", str(script)],
        cwd=str(adslab_dir),
        check=False,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return False, f"sbatch_failed: {proc.stderr.strip() or proc.stdout.strip()}"

    submitted[key] = {"at": _now(), "sbatch_stdout": proc.stdout.strip()}
    _save_json_atomic(state_path, state)
    return True, "submitted"


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare/submit DFT jobs for outputs/*/molecules/*/dft/adslab.")
    ap.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Per-run output root containing molecules/ and slab/, e.g. outputs/adsorption/<timestamp>",
    )
    ap.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="State JSON path (default: <output-root>/dft_adslab_submit_state.json)",
    )
    ap.add_argument("--ncore", type=int, default=16, help="INCAR NCORE value (default: 16)")
    ap.add_argument("--ediffg", type=float, default=-0.02, help="INCAR EDIFFG value (default: -0.02)")
    ap.add_argument(
        "--subvasp-src",
        type=str,
        default=None,
        help="Batch submission script to copy into each adslab dir (required by `prepare`).",
    )
    ap.add_argument(
        "--only",
        action="append",
        default=None,
        help="Optional molecule ID to restrict to (repeatable).",
    )

    sub = ap.add_subparsers(dest="cmd", required=True)
    p_prep = sub.add_parser("prepare", help="Prepare: patch INCAR + copy subvasp.sh")
    p_prep.add_argument("--limit", type=int, default=None, help="Optional limit of directories to process.")

    p_sub = sub.add_parser("submit", help="Submit: sbatch subvasp.sh in prepared dirs")
    p_sub.add_argument("--limit", type=int, default=None, help="Optional limit of directories to process.")

    args = ap.parse_args()

    output_root = Path(args.output_root).resolve()
    state_path = Path(args.state_file).resolve() if args.state_file else output_root / "dft_adslab_submit_state.json"
    state = _load_json(state_path)
    state.setdefault("prepared", {})
    state.setdefault("submitted", {})

    adslab_dirs = _iter_adslab_dirs(output_root)
    if args.only is not None:
        allowed = set(args.only)
        adslab_dirs = [p for p in adslab_dirs if p.parent.parent.name in allowed]  # .../molecules/<sid>/dft/adslab

    if args.cmd == "prepare":
        if not args.subvasp_src:
            ap.error("prepare requires --subvasp-src (the batch submission script to copy into each adslab dir)")
        subvasp_src = Path(args.subvasp_src).expanduser().resolve()
        results: list[PrepareResult] = []
        for i, d in enumerate(adslab_dirs):
            if args.limit is not None and i >= int(args.limit):
                break
            r = prepare_one(
                d,
                ncore=args.ncore,
                ediffg=args.ediffg,
                subvasp_src=subvasp_src,
                state=state,
                state_path=state_path,
            )
            results.append(r)

        prepared_n = sum(1 for r in results if r.prepared)
        skipped_n = len(results) - prepared_n
        print(f"[Prepare] scanned={len(results)} prepared={prepared_n} skipped={skipped_n}")
        # Print a compact skip-reason summary
        reasons: dict[str, int] = {}
        for r in results:
            if not r.prepared:
                reasons[r.reason] = reasons.get(r.reason, 0) + 1
        if reasons:
            print("[Prepare] skip_reasons:", ", ".join(f"{k}={v}" for k, v in sorted(reasons.items())))
        print(f"[Prepare] state_file={state_path}")
        return 0

    if args.cmd == "submit":
        submitted_n = 0
        skipped_n = 0
        for i, d in enumerate(adslab_dirs):
            if args.limit is not None and i >= int(args.limit):
                break
            ok, reason = submit_one(d, state=state, state_path=state_path)
            if ok:
                submitted_n += 1
            else:
                skipped_n += 1
        print(
            f"[Submit] scanned={min(len(adslab_dirs), (args.limit or len(adslab_dirs)))} submitted={submitted_n} skipped={skipped_n}"
        )
        print(f"[Submit] state_file={state_path}")
        return 0

    raise SystemExit("unreachable")


if __name__ == "__main__":
    raise SystemExit(main())
