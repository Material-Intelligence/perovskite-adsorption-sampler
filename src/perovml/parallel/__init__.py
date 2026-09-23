"""Slurm task sharding and result merging.

The modules here read the SLURM environment directly and slice a task list across
ranks, which is enough for the embarrassingly parallel "one molecule per task"
workloads in this repository.
"""

from perovml.parallel.reduce import merge_json_results, reduce_results, verify_completeness
from perovml.parallel.shard import SimpleParallelRunner, get_slurm_env, shard_tasks

__all__ = [
    "SimpleParallelRunner",
    "get_slurm_env",
    "merge_json_results",
    "reduce_results",
    "shard_tasks",
    "verify_completeness",
]
