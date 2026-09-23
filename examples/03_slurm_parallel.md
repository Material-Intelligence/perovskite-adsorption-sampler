# Running in parallel under Slurm

Screening a few hundred molecules is embarrassingly parallel: each molecule is independent, and the
only shared input is one relaxed slab. This package exploits that in the simplest way available —
every rank is an ordinary Python process that reads its own index from the environment and takes a
slice of the molecule list. There is no MPI communicator, no `torch.distributed`, and no launcher
beyond `srun`.

Sizing, file naming and the sharding guarantees are in
[docs/architecture.md](../docs/architecture.md#parallel-execution).

**The job scripts here are templates.** Account, partition, QOS and GPU-request syntax differ
between sites, sometimes in the directive names themselves. Replace every `<PLACEHOLDER>` and check
the directives against your own site's documentation before submitting.

## The model in one paragraph

`shard_tasks` splits the sorted molecule list with `numpy.array_split` at index
`SLURM_PROCID` out of `SLURM_NTASKS`. The slices are disjoint and cover everything, so no
coordination is needed at run time. Each rank writes `<prefix>_<ntasks>-<procid>.json` into the same
output directory, and merging happens afterwards from those files. Outside Slurm the variables are
absent, the code reads back as rank 0 of 1, and the identical script runs as a single process on a
laptop.

## Interactive first

Ask for a short interactive allocation and run one rank before you queue anything long:

```bash
salloc --nodes=1 --ntasks=4 --gpus-per-task=1 --time=00:30:00 \
       --account=<YOUR_ACCOUNT> --partition=<YOUR_GPU_PARTITION>
```

```bash
export DPA3_MODEL_PATH=<PATH_TO_YOUR_CHECKPOINT>/DPA-3.1-3M.pt

srun -n 4 --gpus-per-task=1 \
    python scripts/run_adsorption.py --config examples/configs/dpa3.yaml
```

Each rank prints its own identity and its share of the work, tagged `[Rank N]`, so the first few
lines already tell you whether the split is what you expected.

## A batch job

`examples/slurm/optimize_molecules.sbatch` is a complete template for the molecule-relaxation step.
The screening job has the same shape:

```bash
#!/bin/bash
#SBATCH --job-name=adsorption_screen
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_GPU_PARTITION>
#SBATCH --time=04:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/screen_%j.out

set -euo pipefail
mkdir -p logs

# Activate the environment that has perovml installed, e.g.
#   source <PATH_TO_YOUR_VENV>/bin/activate

export DPA3_MODEL_PATH=<PATH_TO_YOUR_CHECKPOINT>/DPA-3.1-3M.pt

srun -n "${SLURM_NTASKS}" --gpus-per-task=1 \
    python scripts/run_adsorption.py \
    --config examples/configs/dpa3.yaml \
    --output outputs/screen_run
```

`--output` pointing at a fixed directory is deliberate: re-submitting the same script is how you
resume. Nothing is timestamped behind your back, and molecules that finished are skipped.

## Choosing the number of ranks

`srun -n N` is the only place the shard count is set; there is no second setting to keep in step
with it.

- **Ranks ≤ molecules.** Surplus ranks get empty slices and sit idle holding GPUs.
- **One GPU per rank.** Each rank loads its own copy of the potential, so memory scales with ranks
  per node, not with the molecule count.
- **Prefer more, shorter jobs over one long one.** The resume is per molecule, so a job that dies at
  the wall clock loses only the molecules in flight. Two two-hour jobs lose less work to a wall-clock
  kill than one four-hour job, and often start sooner.
- **Uneven work?** `array_split` gives the remainder to the lowest-numbered ranks, so rank 0 finishes
  last. With molecules of very different sizes, use more shards than nodes and let the resume absorb
  the ragged edge.

## After the job

Each rank leaves its own result file. Merge them and check that the run really covered everything —
a count of output directories will not reveal a rank that died before writing:

```python
from perovml.parallel import merge_json_results, verify_completeness

records = merge_json_results("outputs/screen_run", pattern="adsorption_*-*.json")
report = verify_completeness("outputs/screen_run", expected_tasks=240,
                             pattern="adsorption_*-*.json")
print(report)
```

On a one-molecule, one-rank smoke test the same call returns:

```python
{'files_found': 1, 'expected_num_jobs': 1, 'missing_jobs': [], 'total_records': 1,
 'failed_records': 0, 'failed_tasks': [], 'expected_tasks': 1, 'is_complete': True}
```

A non-empty `missing_jobs` names the ranks whose files never appeared. Re-submit the same script:
completed molecules are skipped and only the lost work is redone.

`scripts/run_parallel_simple.py --reduce-only` does the merge in one command for runs made with that
script, and `perovml.parallel.reduce_results` returns a pandas `DataFrame` while writing
`combined.csv` and `combined.json` beside the shard files.

## Pitfalls

- **Do not nest `srun`.** When the DFT stage runs inside a job that is already under `srun`, set
  `vasp_command: vasp_std`, not `srun vasp_std`.
- **Check that shared paths are visible from the compute node.** `slab_dft_dir`, the model
  checkpoint and the output directory all have to be readable there, not just from the login node.
- **Give the output directory to every rank identically.** Ranks find each other's work only through
  that directory; if `--output` differs between ranks, the resume and the completeness check both
  break.
- **Keep the molecule directory stable during a run.** The task list comes from a sorted glob
  evaluated independently in each rank, so adding a file mid-run changes the split underneath the
  ranks that have not started yet.
- **Redirect logs per job.** Four ranks writing to one terminal interleave; `%j` in the `--output`
  directive keeps runs apart, and each line is already prefixed with its rank.
