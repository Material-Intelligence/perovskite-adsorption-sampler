"""Aggregation of the per-rank result files written by :mod:`perovml.parallel.shard`."""

from __future__ import annotations

import gzip
import json
import logging
from glob import glob
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only
    import pandas as pd

logger = logging.getLogger(__name__)

__all__ = ["merge_json_results", "reduce_results", "verify_completeness"]

DEFAULT_RESULT_PATTERN = "result_*-*.json*"
"""Glob matching the per-rank files ``SimpleParallelRunner`` writes."""


def _read_json(path: str | Path) -> Any:
    """Read a JSON file, transparently handling gzip compression.

    Args:
        path: Path to a ``.json`` or ``.json.gz`` file.

    Returns:
        The decoded JSON content.
    """
    path = Path(path)
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as file:
            return json.load(file)
    with open(path) as file:
        return json.load(file)


def merge_json_results(
    results_dir: str | Path,
    pattern: str = DEFAULT_RESULT_PATTERN,
    output_file: str | Path | None = None,
) -> list[Any]:
    """Concatenate every per-rank JSON result file in a directory.

    Args:
        results_dir: Directory holding the per-rank files.
        pattern: Glob matching those files.
        output_file: Optional path to write the merged list to. A ``.gz``
            suffix selects gzip output.

    Returns:
        The merged records, in file-name order. Empty if nothing matched.
    """
    files = sorted(glob(str(Path(results_dir) / pattern)))

    if not files:
        logger.warning("no result file matched %s/%s", results_dir, pattern)
        return []

    all_results: list[Any] = []
    for path in files:
        data = _read_json(path)
        if isinstance(data, list):
            all_results.extend(data)
        else:
            all_results.append(data)

    logger.info("merged %d files, %d records total", len(files), len(all_results))

    if output_file:
        out_path = Path(output_file)
        if out_path.suffix == ".gz":
            with gzip.open(out_path, "wt", encoding="utf-8") as file:
                json.dump(all_results, file, indent=2, default=str)
        else:
            with open(out_path, "w") as file:
                json.dump(all_results, file, indent=2, default=str)
        logger.info("merged results written to %s", output_file)

    return all_results


def reduce_results(
    results_dir: str | Path,
    pattern: str = DEFAULT_RESULT_PATTERN,
    output_prefix: str = "combined",
) -> pd.DataFrame:
    """Merge the per-rank results and write them as CSV and JSON.

    Args:
        results_dir: Directory holding the per-rank files; the combined files are
            written there too.
        pattern: Glob matching the per-rank files.
        output_prefix: Stem of the combined ``.csv`` and ``.json`` files.

    Returns:
        The merged records as a DataFrame, empty if nothing matched.
    """
    import pandas as pd

    all_results = merge_json_results(results_dir, pattern)

    if not all_results:
        return pd.DataFrame()

    dataframe = pd.DataFrame(all_results)

    out_dir = Path(results_dir)
    dataframe.to_csv(out_dir / f"{output_prefix}.csv", index=False)
    dataframe.to_json(out_dir / f"{output_prefix}.json", orient="records", indent=2)

    logger.info("combined table written to %s/%s.{csv,json}", out_dir, output_prefix)

    return dataframe


def verify_completeness(
    results_dir: str | Path,
    expected_tasks: int,
    pattern: str = DEFAULT_RESULT_PATTERN,
) -> dict[str, Any]:
    """Check that every rank wrote its file and that every task actually succeeded.

    The number of ranks is inferred from the ``<prefix>_<num_jobs>-<job_num>``
    part of the file names, so no SLURM environment is needed.

    A task that raised is still written out, as ``{"task": ..., "error": ...}``, so
    counting records alone would report a run in which every single molecule crashed as
    complete. Only records without an ``error`` key count towards ``total_records``, and
    ``is_complete`` additionally requires that none failed.

    Args:
        results_dir: Directory holding the per-rank files.
        expected_tasks: Number of records the complete run should contain.
        pattern: Glob matching the per-rank files.

    Returns:
        A report with the keys ``files_found``, ``expected_num_jobs``,
        ``missing_jobs``, ``total_records`` (successful ones only),
        ``failed_records``, ``failed_tasks``, ``expected_tasks`` and ``is_complete``.
    """
    files = sorted(glob(str(Path(results_dir) / pattern)))

    num_jobs_set: set[int] = set()
    job_nums: set[int] = set()
    for path in files:
        name = Path(path).stem.replace(".json", "")  # strips the extra stem of a .json.gz name
        parts = name.split("_")[-1].split("-")
        if len(parts) == 2:
            num_jobs_set.add(int(parts[0]))
            job_nums.add(int(parts[1]))

    expected_num_jobs = max(num_jobs_set) if num_jobs_set else 0
    missing_jobs = set(range(expected_num_jobs)) - job_nums

    total_records = 0
    failed_tasks: list[str] = []
    for path in files:
        data = _read_json(path)
        records = data if isinstance(data, list) else [data]
        for record in records:
            if isinstance(record, dict) and record.get("error") is not None:
                failed_tasks.append(str(record.get("task", record.get("sid", "<unnamed task>"))))
            else:
                total_records += 1

    report: dict[str, Any] = {
        "files_found": len(files),
        "expected_num_jobs": expected_num_jobs,
        "missing_jobs": sorted(missing_jobs),
        "total_records": total_records,
        "failed_records": len(failed_tasks),
        "failed_tasks": failed_tasks,
        "expected_tasks": expected_tasks,
        "is_complete": (len(missing_jobs) == 0 and not failed_tasks and total_records == expected_tasks),
    }

    if report["is_complete"]:
        logger.info("verification passed: %d/%d tasks complete", total_records, expected_tasks)
    else:
        logger.warning(
            "verification failed: missing ranks %s, %d/%d records succeeded, %d failed%s",
            sorted(missing_jobs),
            total_records,
            expected_tasks,
            len(failed_tasks),
            f" ({', '.join(failed_tasks[:5])}{', ...' if len(failed_tasks) > 5 else ''})" if failed_tasks else "",
        )

    return report
