"""Tests for the completeness check that decides whether a campaign is finished.

A screening run is declared complete on the strength of this function, so its failure
mode matters: it is asked about a directory of per-rank files, and the records inside can
be either a finished molecule or a task that raised.
"""

from __future__ import annotations

import json
from pathlib import Path

from perovml.parallel.reduce import verify_completeness


def _write_rank(results_dir: Path, num_jobs: int, job_num: int, records: list[dict]) -> None:
    """Write one rank's result file, named the way SimpleParallelRunner names them."""
    results_dir.mkdir(parents=True, exist_ok=True)
    (results_dir / f"result_{num_jobs}-{job_num}.json").write_text(json.dumps(records))


class TestVerifyCompleteness:
    def test_a_finished_run_is_complete(self, tmp_path):
        _write_rank(tmp_path, 2, 0, [{"sid": "mol_a", "best_energy": -1.0}])
        _write_rank(tmp_path, 2, 1, [{"sid": "mol_b", "best_energy": -2.0}])

        report = verify_completeness(tmp_path, expected_tasks=2)

        assert report["is_complete"] is True
        assert report["total_records"] == 2
        assert report["failed_records"] == 0
        assert report["missing_jobs"] == []

    def test_a_run_in_which_every_task_crashed_is_not_complete(self, tmp_path):
        """The record count alone cannot tell this run apart from a successful one."""
        _write_rank(tmp_path, 2, 0, [{"task": "mol_a", "error": "CUDA out of memory"}])
        _write_rank(tmp_path, 2, 1, [{"task": "mol_b", "error": "CUDA out of memory"}])

        report = verify_completeness(tmp_path, expected_tasks=2)

        assert report["is_complete"] is False
        assert report["total_records"] == 0
        assert report["failed_records"] == 2
        assert report["failed_tasks"] == ["mol_a", "mol_b"]

    def test_one_failure_among_many_is_named(self, tmp_path):
        _write_rank(
            tmp_path,
            1,
            0,
            [
                {"sid": "mol_a", "best_energy": -1.0},
                {"task": "mol_b", "error": "structure could not be read"},
                {"sid": "mol_c", "best_energy": -3.0},
            ],
        )

        report = verify_completeness(tmp_path, expected_tasks=3)

        assert report["is_complete"] is False
        assert report["total_records"] == 2
        assert report["failed_tasks"] == ["mol_b"]

    def test_a_missing_rank_is_reported(self, tmp_path):
        _write_rank(tmp_path, 3, 0, [{"sid": "mol_a"}])
        _write_rank(tmp_path, 3, 2, [{"sid": "mol_c"}])

        report = verify_completeness(tmp_path, expected_tasks=3)

        assert report["is_complete"] is False
        assert report["missing_jobs"] == [1]
