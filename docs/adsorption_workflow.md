# The MLP-to-DFT adsorption workflow

Screening a directory of molecules against one slab, in two stages: a cheap machine-learned-potential
(MLP) stage that generates and relaxes many candidate configurations, and an optional DFT stage that
takes the survivors to VASP. It restarts at the granularity of a single configuration, so a run
killed at its wall-clock limit loses only the structures in flight.

The entry point is `scripts/run_adsorption.py`; the per-molecule logic lives in
`perovml.recipes.adsorption.run_adsorption_task`.

## Stages

For each molecule, in order:

1. **Slab reference.** The bare slab is relaxed once, before the molecule loop, and the relaxed
   structure and energy are handed to every task. With `slab_mlp_relaxed: true` it is treated as
   already relaxed and only a single point is done.
2. **Gas-phase reference.** The isolated molecule is relaxed. Its energy is the second term of
   E_ads = E_adslab − E_slab − E_molecule.
3. **Dual sampling.** `Pb_uniform_sample` and `Pb_heuristic_sample` both run
   ([pb_sampling_algorithm.md](pb_sampling_algorithm.md)); the two sets are merged and written to
   `configs/` as `uniform_NNN.vasp` and `heuristic_NNN.vasp`, each with a `_tags.json` sidecar that
   preserves the bulk/surface/adsorbate tags a POSCAR cannot carry.
4. **Relaxation, one configuration at a time.** Every finished configuration appends itself to
   `status.json` before the next one starts. This is the granularity the resume works at.
5. **Ranking.** Anomaly severity first (no anomaly, then desorption only, then more serious
   anomalies), adsorption energy within a severity class. The best `save_top` are kept; the single
   best is also written as `mlp/best.vasp`.
6. **DFT (optional).** Molecule, then slab, then the best adslab. See
   [vasp_stage.md](vasp_stage.md).

## Running it

```bash
python scripts/run_adsorption.py --config examples/configs/dpa3.yaml
```

On a cluster, one Slurm task per GPU; the molecule list is sharded across ranks by `SLURM_PROCID`
and `SLURM_NTASKS`, with no MPI communicator involved
([architecture.md](architecture.md#parallel-execution)):

```bash
srun -n 16 --gpus-per-task=1 python scripts/run_adsorption.py \
    --config examples/configs/dpa3.yaml
```

Command-line flags override the config: `--slab`, `--molecules-dir`, `--output`, and `--no-dft` to
force `run_dft: false` for one invocation. Every configuration key is in
[parameters.md](parameters.md#3-batch-workflow).

To resume, run the same command again. The output directory is used exactly as configured — no
timestamp is appended — so re-running with the same `output` is what continues the job. An existing
`metadata.json` is the marker:

```
[Info] resuming run: outputs/adsorption
[Info] 1 tasks, 1 done, 0 to run
[Info] every task is already finished
```

Molecules whose `status.json` shows the required stages complete are skipped outright. A molecule
interrupted mid-relaxation restarts from its last finished configuration and re-reads the
configurations it had already generated from `mlp/configs.traj` rather than sampling again — the
sampling is random, so re-sampling would silently change what was being screened. The `.vasp` files
under `configs/` are copies for inspection; neither the resume nor the anomaly detection reads them,
because a POSCAR cannot carry the tags and constraints they depend on.

## Task directory layout

```
outputs/adsorption/
├── metadata.json                     # config snapshot, task count, Slurm job id
├── adsorption_<ntasks>-<rank>.json   # one result file per rank
├── slab/
│   ├── slab_opt.vasp                 # shared relaxed slab
│   └── slab_opt.json
└── molecules/<molecule>/
    ├── config.json                   # frozen parameters for this task
    ├── status.json                   # progress and results
    ├── configs/
    │   ├── uniform_000.vasp
    │   ├── uniform_000_tags.json
    │   ├── heuristic_000.vasp
    │   └── ...
    ├── mlp/
    │   ├── molecule.vasp  molecule.json    # relaxed gas-phase reference
    │   ├── slab.vasp      slab.json        # slab reference as used here
    │   ├── configs.traj   results.traj     # generated and relaxed trajectories
    │   ├── results/<name>.vasp, <name>.json
    │   ├── best.vasp  best_tags.json
    │   └── all_results.json
    └── dft/                                # only when run_dft or dft_generate_only
        ├── molecule/                       # INCAR POSCAR KPOINTS POTCAR (+ outputs)
        ├── slab/                           # a symlink when slab_dft_dir is set
        └── adslab/
```

A task directory is self-contained. `config.json` holds the parameters the task was started with,
so a resume never inherits a changed setting by accident, and `status.json` is the single source of
truth for what has been done:

```json
{
  "slab_optimized": true,
  "slab_energy": 440.81307518298416,
  "molecule_optimized": true,
  "molecule_energy": 116.61141907182515,
  "configs_generated": true,
  "config_names": ["uniform_000", "uniform_001", "heuristic_000", "heuristic_001"],
  "configs_completed": ["heuristic_001", "uniform_000", "heuristic_000", "uniform_001"],
  "best_config": "uniform_001",
  "best_energy": 475.8187410577983,
  "best_adsorption_energy": -81.605753197011,
  "best_anomalies": [],
  "best_converged": true,
  "best_fmax": 0.196984338231996,
  "best_nsteps": 0,
  "configs_total": 4,
  "configs_converged": 4,
  "mlp_done": true
}
```

(The energies above come from a mock-calculator smoke test and are not physical.)

`configs_completed` is unordered — it records completion, not rank — and the DFT stage adds its own
flags to the same file: `dft_molecule_done`, `dft_slab_done`, `dft_adslab_done`, with the matching
energies.

## Sharing a slab DFT calculation

The bare slab is the same for every molecule, and its DFT calculation is often the most expensive
single item in the campaign. Point `slab_dft_dir` at a finished slab directory and each task
symlinks its own `dft/slab` to it and reads the energy from that `OUTCAR` instead of recomputing:

```yaml
slab_dft_dir: ../shared/slab_scf
```

The path must be readable from the compute node. If it is set but missing, the task raises rather
than quietly recomputing. `dft_slab_energy` does the same job when you only have the number.

## Generating inputs without running VASP

`dft_generate_only: true` writes `INCAR`, `POSCAR`, `KPOINTS` and `POTCAR` into each `dft/`
subdirectory and stops. Useful when VASP runs under a different account, a different queue, or a
different machine from the MLP screening. The workflow records that the inputs exist, so a later run
with `dft_generate_only: false` picks up from there.

## Checking a finished run

Each rank writes `adsorption_<ntasks>-<rank>.json`. Merge them, and check that the run actually
covered every molecule, with the helpers in `perovml.parallel`:

```python
from perovml.parallel import merge_json_results, verify_completeness

records = merge_json_results("outputs/adsorption", pattern="adsorption_*-*.json")
report = verify_completeness("outputs/adsorption", expected_tasks=240,
                             pattern="adsorption_*-*.json")
```

`verify_completeness` returns a dict with `files_found`, `missing_jobs`, `total_records`,
`failed_records`, `failed_tasks` and `is_complete`. A missing rank means a task died without
writing, which a bare count of output directories would not reveal.

A task that raised is still written out, as `{"task": ..., "error": ...}`, so a run in which every
molecule crashed produces exactly as many records as a successful one. Only records without an
`error` count towards `total_records`, the failures are counted in `failed_records` and named in
`failed_tasks`, and `is_complete` requires that there are none.
