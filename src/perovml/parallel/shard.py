"""Task sharding across SLURM ranks.

The module reads the SLURM environment, splits a task list with
:func:`numpy.array_split` so that the shards are disjoint and cover every task,
and wraps the result in a small runner that writes one JSON file per rank.

Examples:
    Launch one shard per process and let each of them pick up its own slice::

        srun -n 8 --gpus-per-task=1 python my_script.py
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, TypeVar

import numpy as np

logger = logging.getLogger(__name__)

T = TypeVar("T")

__all__ = ["ParallelConfig", "SimpleParallelRunner", "SlurmEnv", "get_slurm_env", "shard_tasks"]


@dataclass
class SlurmEnv:
    """The subset of the SLURM environment this package needs.

    Attributes:
        job_num: Index of this process, from ``SLURM_PROCID``.
        num_jobs: Total number of processes, from ``SLURM_NTASKS``.
        local_rank: Index of this process within its node, from ``SLURM_LOCALID``.
        job_id: SLURM job id, from ``SLURM_JOB_ID``.
        nodelist: Allocated node list, from ``SLURM_NODELIST``.
        is_slurm: Whether the process appears to run under SLURM at all.
    """

    job_num: int = 0
    num_jobs: int = 1
    local_rank: int = 0
    job_id: str = ""
    nodelist: str = ""
    is_slurm: bool = False

    def as_dict(self) -> dict[str, Any]:
        """Return an MSONable-style dict representation.

        Returns:
            A JSON-serialisable dict carrying ``@module``/``@class`` keys alongside the fields.
        """
        return {"@module": type(self).__module__, "@class": type(self).__name__, **asdict(self)}

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> SlurmEnv:
        """Reconstruct an environment record from :meth:`as_dict` output.

        Args:
            dct: A dict produced by :meth:`as_dict`.

        Returns:
            The reconstructed :class:`SlurmEnv`.
        """
        fields = {key: value for key, value in dct.items() if not key.startswith("@")}
        return cls(**fields)


def get_slurm_env() -> SlurmEnv:
    """Read the SLURM rank and job information from the environment.

    Returns:
        A :class:`SlurmEnv`. Outside SLURM it describes a single-process run.
    """
    is_slurm = "SLURM_PROCID" in os.environ or "SLURM_JOB_ID" in os.environ

    return SlurmEnv(
        job_num=int(os.environ.get("SLURM_PROCID", 0)),
        num_jobs=int(os.environ.get("SLURM_NTASKS", 1)),
        local_rank=int(os.environ.get("SLURM_LOCALID", 0)),
        job_id=os.environ.get("SLURM_JOB_ID", ""),
        nodelist=os.environ.get("SLURM_NODELIST", ""),
        is_slurm=is_slurm,
    )


def shard_tasks(tasks: list[T], job_num: int, num_jobs: int) -> list[T]:
    """Split a task list and return the slice this process owns.

    :func:`numpy.array_split` guarantees that the shards do not overlap, that
    together they cover every task, and that their sizes differ by at most one.

    Args:
        tasks: The complete task list.
        job_num: Index of this process, 0-based.
        num_jobs: Total number of processes.

    Returns:
        The sub-list of tasks this process is responsible for.

    Examples:
        >>> tasks = list(range(10))
        >>> shard_tasks(tasks, job_num=0, num_jobs=3)
        [0, 1, 2, 3]
        >>> shard_tasks(tasks, job_num=1, num_jobs=3)
        [4, 5, 6]
        >>> shard_tasks(tasks, job_num=2, num_jobs=3)
        [7, 8, 9]
    """
    if num_jobs <= 1:
        return tasks

    indices = np.array_split(np.arange(len(tasks)), num_jobs)[job_num]
    return [tasks[i] for i in indices]


@dataclass
class ParallelConfig:
    """Output settings for a sharded run.

    Attributes:
        output_dir: Directory the per-rank result files are written to.
        result_prefix: Stem of the per-rank result file names.
        save_format: Either ``"json"`` or ``"json.gz"``.
        tasks_glob: Optional glob describing where the tasks come from, e.g.
            ``"data/molecules/*.vasp"``. Informational only.
    """

    output_dir: str = "outputs"
    result_prefix: str = "result"
    save_format: str = "json"
    tasks_glob: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> ParallelConfig:
        """Load a configuration from a YAML file, ignoring unknown keys.

        Args:
            path: Path to the YAML file.

        Returns:
            The parsed configuration.
        """
        import yaml

        with open(path) as file:
            cfg = yaml.safe_load(file) or {}
        return cls(**{key: value for key, value in cfg.items() if hasattr(cls, key)})

    def as_dict(self) -> dict[str, Any]:
        """Return an MSONable-style dict representation.

        Returns:
            A JSON-serialisable dict carrying ``@module``/``@class`` keys alongside the fields.
        """
        return {"@module": type(self).__module__, "@class": type(self).__name__, **asdict(self)}

    @classmethod
    def from_dict(cls, dct: dict[str, Any]) -> ParallelConfig:
        """Reconstruct a configuration from :meth:`as_dict` output.

        Args:
            dct: A dict produced by :meth:`as_dict`.

        Returns:
            The reconstructed :class:`ParallelConfig`.
        """
        fields = {key: value for key, value in dct.items() if not key.startswith("@")}
        return cls(**fields)


class SimpleParallelRunner:
    """Run one shard of a task list and write this rank's results.

    The runner reads the SLURM environment on construction, so the same script
    works unchanged on one process and on many.

    Examples:
        >>> runner = SimpleParallelRunner(output_dir="outputs/my_run")  # doctest: +SKIP
        >>> my_tasks = runner.get_my_tasks(sorted(glob("data/molecules/*.vasp")))  # doctest: +SKIP
        >>> runner.save_results([process(task) for task in my_tasks])  # doctest: +SKIP

    Attributes:
        config: Output settings for this run.
        slurm: The SLURM environment this process sees.
        output_path: Directory the result files are written to.
    """

    def __init__(
        self,
        output_dir: str = "outputs",
        result_prefix: str = "result",
        config: ParallelConfig | None = None,
    ) -> None:
        """Set up the runner and create its output directory.

        Args:
            output_dir: Directory for the result files. Ignored when ``config`` is given.
            result_prefix: Stem of the result file names. Ignored when ``config`` is given.
            config: A fully built configuration, taking precedence over the two
                arguments above.
        """
        self.config = config or ParallelConfig(output_dir=output_dir, result_prefix=result_prefix)
        self.slurm = get_slurm_env()

        self.output_path = Path(self.config.output_dir)
        self.output_path.mkdir(parents=True, exist_ok=True)

        self._log(f"initialised | SLURM={self.slurm.is_slurm} | rank={self.slurm.job_num}/{self.slurm.num_jobs}")

    def _log(self, msg: str) -> None:
        """Emit a message tagged with this process's rank.

        Args:
            msg: Message body.
        """
        logger.info("[Rank %d] %s", self.slurm.job_num, msg)

    @property
    def job_num(self) -> int:
        """Index of this process, 0-based."""
        return self.slurm.job_num

    @property
    def num_jobs(self) -> int:
        """Total number of processes."""
        return self.slurm.num_jobs

    @property
    def is_main(self) -> bool:
        """Whether this is the main process (rank 0)."""
        return self.slurm.job_num == 0

    def get_my_tasks(self, all_tasks: list[T]) -> list[T]:
        """Return the shard of ``all_tasks`` this process owns.

        Args:
            all_tasks: The complete task list, identical on every rank.

        Returns:
            This process's sub-list of tasks.
        """
        my_tasks = shard_tasks(all_tasks, self.job_num, self.num_jobs)
        self._log(f"task split: {len(my_tasks)}/{len(all_tasks)}")
        return my_tasks

    def run(
        self,
        tasks: list[T],
        process_fn: Callable[[T], Any],
        desc: str = "Processing",
    ) -> list[Any]:
        """Shard the task list, process this rank's slice and save the results.

        A task that raises is recorded as ``{"task": ..., "error": ...}`` instead
        of aborting the run.

        Args:
            tasks: The complete task list; it is sharded internally.
            process_fn: Callable applied to a single task.
            desc: Progress-bar description.

        Returns:
            This process's result list.
        """
        my_tasks = self.get_my_tasks(tasks)

        results: list[Any] = []
        try:
            from tqdm import tqdm

            iterator: Any = tqdm(my_tasks, desc=f"[R{self.job_num}] {desc}")
        except ImportError:
            iterator = my_tasks

        for task in iterator:
            try:
                results.append(process_fn(task))
            except Exception as exc:  # noqa: BLE001 - one failed task must not kill the shard
                self._log(f"task failed: {task} - {exc}")
                results.append({"task": str(task), "error": str(exc)})

        self.save_results(results)
        return results

    def get_result_path(self) -> Path:
        """Return the result file path for this process.

        Returns:
            ``<output_dir>/<prefix>_<num_jobs>-<job_num>.json[.gz]``.
        """
        suffix = ".json.gz" if self.config.save_format == "json.gz" else ".json"
        filename = f"{self.config.result_prefix}_{self.num_jobs}-{self.job_num}{suffix}"
        return self.output_path / filename

    def save_results(self, results: list[Any]) -> Path:
        """Write this process's results to :meth:`get_result_path`.

        Args:
            results: Records to serialise. Values that JSON cannot represent are
                written with ``str()``.

        Returns:
            The path written to.
        """
        out_path = self.get_result_path()

        if out_path.suffix == ".gz":
            import gzip

            with gzip.open(out_path, "wt", encoding="utf-8") as file:
                json.dump(results, file, indent=2, default=str)
        else:
            with open(out_path, "w") as file:
                json.dump(results, file, indent=2, default=str)

        self._log(f"results written: {out_path} ({len(results)} records)")
        return out_path

    def save_metadata(self, metadata: dict[str, Any]) -> Path:
        """Write ``metadata.json`` describing the run. Main process only.

        The timestamp, SLURM job id and process count are added automatically.

        Args:
            metadata: Caller-supplied fields, typically the run configuration.

        Returns:
            The path written to, or an empty :class:`~pathlib.Path` on other ranks.
        """
        if not self.is_main:
            return Path()

        meta_path = self.output_path / "metadata.json"
        metadata.update(
            {
                "timestamp": datetime.now().isoformat(),
                "slurm_job_id": self.slurm.job_id,
                "num_jobs": self.num_jobs,
            }
        )
        with open(meta_path, "w") as file:
            json.dump(metadata, file, indent=2)
        return meta_path
