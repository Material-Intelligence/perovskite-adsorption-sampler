#!/bin/bash
# Batch relaxation of the probe molecules with an MLP, one Slurm task per GPU.
#
# This is a template, not a ready-to-submit job: replace every <PLACEHOLDER> below and
# check the directive names against your own site's documentation, because account,
# partition, QOS and GPU-request syntax differ from cluster to cluster.
#
#   1. Edit the #SBATCH directives and the environment-activation line.
#   2. Submit from the repository root:
#
#        sbatch examples/slurm/optimize_molecules.sbatch
#
# scripts/optimize_molecules.py shards the molecule list across ranks using SLURM_PROCID
# and SLURM_NTASKS, so it needs no MPI communicator; mpi4py is used only when it is
# importable and Slurm variables are absent. Re-submitting with --resume skips molecules
# that already have an output structure.
#
# examples/03_slurm_parallel.md explains the sharding model, how to size -n, and how to
# check afterwards that every task was covered. The screening job has the same shape.

#SBATCH --job-name=optimize_molecules
#SBATCH --account=<YOUR_ACCOUNT>
#SBATCH --partition=<YOUR_GPU_PARTITION>
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gpus-per-node=4
#SBATCH --output=logs/optimize_molecules_%j.out
#SBATCH --error=logs/optimize_molecules_%j.err

set -euo pipefail

mkdir -p logs

# Activate the environment that has perovml installed, for example:
#   source <PATH_TO_YOUR_VENV>/bin/activate
# Anything that puts `python` and the perovml package on PATH will do.

# DPA-3 checkpoint. The weights are not shipped with this repository; see models/README.md
# for where to download them.
export DPA3_MODEL_PATH="${DPA3_MODEL_PATH:-<PATH_TO_DPA3_CHECKPOINT>}"

srun -n "${SLURM_NTASKS}" --gpus-per-task=1 \
    python scripts/optimize_molecules.py \
    --molecules-dir data/molecules \
    --output outputs/optimized_molecules \
    --calculator dpa3 \
    --model-path "${DPA3_MODEL_PATH}" \
    --fmax 0.01 \
    --steps 200 \
    --resume
