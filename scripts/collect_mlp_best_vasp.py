#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shutil
import zipfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


def _now_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


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


@dataclass(frozen=True)
class Candidate:
    sid: str
    best_vasp: Path


def _iter_mlp_done_best(output_root: Path) -> list[Candidate]:
    mols = output_root / "molecules"
    if not mols.exists():
        return []
    out: list[Candidate] = []
    for mol_dir in sorted(mols.iterdir()):
        if not mol_dir.is_dir():
            continue
        sid = mol_dir.name
        status = _load_json(mol_dir / "status.json")
        if not status.get("mlp_done", False):
            continue
        best = mol_dir / "mlp" / "best.vasp"
        if best.exists():
            out.append(Candidate(sid=sid, best_vasp=best))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Collect outputs/<run>/molecules/*/mlp/best.vasp for mlp_done tasks into a new per-run folder."
    )
    ap.add_argument(
        "--output-root",
        type=str,
        required=True,
        help="Per-run output root containing molecules/, e.g. outputs/adsorption/<timestamp>",
    )
    ap.add_argument(
        "--dest-root",
        type=str,
        default=None,
        help="Destination root (default: <output-root>/collected_best_vasp)",
    )
    ap.add_argument(
        "--state-file",
        type=str,
        default=None,
        help="State JSON path (default: <dest-root>/collect_state.json)",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry-run: print what would be collected without copying anything.",
    )
    args = ap.parse_args()

    output_root = Path(args.output_root).resolve()
    dest_root = Path(args.dest_root).resolve() if args.dest_root else (output_root / "collected_best_vasp")
    state_path = Path(args.state_file).resolve() if args.state_file else (dest_root / "collect_state.json")

    state = _load_json(state_path)
    collected: dict[str, Any] = state.setdefault("collected", {})

    candidates = _iter_mlp_done_best(output_root)
    to_collect: list[Candidate] = []
    for c in candidates:
        key = str(c.best_vasp.resolve())
        if key in collected:
            continue
        to_collect.append(c)

    print(f"[Info] output_root={output_root}")
    print(f"[Info] dest_root={dest_root}")
    print(f"[Info] state_file={state_path}")
    print(f"[Info] mlp_done best.vasp found={len(candidates)} new_to_collect={len(to_collect)}")

    run_id = _now_id()
    run_dir = dest_root / f"run_{run_id}"

    if not to_collect:
        if not args.dry_run:
            dest_root.mkdir(parents=True, exist_ok=True)
            # record an empty run for provenance
            runs = state.setdefault("runs", [])
            runs.append({"run_id": run_id, "at": datetime.now().isoformat(timespec="seconds"), "new": 0})
            _save_json_atomic(state_path, state)
        print("[Info] nothing new to collect.")
        return 0

    if args.dry_run:
        for c in to_collect[:20]:
            print(f"[DryRun] {c.sid} <- {c.best_vasp}")
        if len(to_collect) > 20:
            print(f"[DryRun] ... (+{len(to_collect) - 20} more)")
        return 0

    run_dir.mkdir(parents=True, exist_ok=False)

    manifest = []
    for c in to_collect:
        dst = run_dir / f"{c.sid}.vasp"
        if dst.exists():
            raise RuntimeError(f"duplicate sid output in this run: {dst}")
        shutil.copy2(c.best_vasp, dst)
        src_key = str(c.best_vasp.resolve())
        collected[src_key] = {
            "sid": c.sid,
            "src": src_key,
            "dst": str(dst.resolve()),
            "collected_at": datetime.now().isoformat(timespec="seconds"),
        }
        manifest.append({"sid": c.sid, "src": src_key, "dst": str(dst)})

    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")

    zip_path = run_dir.with_suffix(".zip")
    with zipfile.ZipFile(zip_path, mode="w", compression=zipfile.ZIP_DEFLATED) as zf:
        for p in sorted(run_dir.rglob("*")):
            if p.is_dir():
                continue
            zf.write(p, arcname=str(p.relative_to(run_dir.parent)))

    runs = state.setdefault("runs", [])
    runs.append(
        {
            "run_id": run_id,
            "at": datetime.now().isoformat(timespec="seconds"),
            "new": len(to_collect),
            "dir": str(run_dir),
            "zip": str(zip_path),
        }
    )
    _save_json_atomic(state_path, state)

    print(f"[Done] collected={len(to_collect)} into {run_dir}")
    print(f"[Done] zipped to {zip_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
