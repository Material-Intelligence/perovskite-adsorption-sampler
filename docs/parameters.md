# Parameter reference

Three things in this repository read configuration, and they do **not** share a schema:

| Schema | Read by | Section |
|---|---|---|
| `perovml run` config | `perovml run -c CONFIG` | [1](#1-perovml-run) |
| `perovml place` options | `perovml place` flags, or `--config` | [2](#2-perovml-place) |
| Batch-workflow config | `scripts/run_adsorption.py`, `scripts/run_parallel_simple.py` | [3](#3-batch-workflow) |

The most common mistake is feeding one to the other. A `perovml run` config names its input with
`slab_file`; a batch config names it `slab` and takes a whole `molecules_dir`. Keys that a schema
does not know are ignored silently, and a missing required key raises immediately.

---

## 1. `perovml run`

Sample placements on one slab for one adsorbate and relax them. Minimal working config:

```yaml
slab_file: data/slabs/FAPbI3_3x3_zcut017.vasp
molecule_file: data/molecules/Acetone.vasp
generator: Pb_heuristic_sample
pb_num: 6
pb_cone: 30.0
min_ab: 8.0
calculator: MockCalculator
optimizer_cls: LBFGS
fmax: 0.5
steps: 10
```

```bash
perovml run -c my_run.yaml --out-root outputs
```

### Structure input

| Key | Default | Meaning |
|---|---|---|
| `slab_file` | *required* | Slab structure, any ASE-readable format. |
| `molecule_file` | — | Adsorbate structure file. Give this **or** `adsorbate`. |
| `adsorbate` | — | Adsorbate key from the FAIRChem database, e.g. `"*O"`. |
| `binding_index` | guessed | 0-based index of the atom expected to face the surface. When omitted it is inferred and written to `binding_guess.txt` in the run directory. |
| `min_ab` | `8.0` | Minimum *a* and *b* cell length, in Å, that FAIRChem's `Slab` expects. Below it you get a warning, not an error. The shipped 3×3 slabs are 19.4 Å in *a* and 18.8 Å in *b*, comfortably above. |

Raise `min_ab` if your molecule is large enough to see its own periodic image; the cure is a larger
supercell, not a smaller threshold.

### Sampling

| Key | Default | Meaning |
|---|---|---|
| `generator` | `Pb_uniform_sample` | `Pb_uniform_sample`, `Pb_heuristic_sample`, `single` or `multi`. |
| `pb_num` | `20` | Number of orientations for either `Pb_*` generator. |
| `pb_cone` | `20.0` | Cone half-angle in degrees. `Pb_heuristic_sample` only. |
| `mode` | `random` | Placement mode for the FAIRChem samplers (`random`, `heuristic`, `random_site_heuristic_placement`). Ignored by the `Pb_*` generators, which place at a single site by construction. |
| `num_sites` | `100` | Sites to sample, FAIRChem sampler only. |
| `num_augmentations` | `1` | Orientations per site, FAIRChem sampler only. |
| `num_configurations` | `1` | Configurations for the multi-adsorbate sampler. |
| `interstitial_gap` | `0.1` | Extra clearance in Å kept when the molecule is lifted along the surface normal, on top of the covalent-radii contact distance. `Pb_*` generators only. |
| `seed` | drawn and recorded | Seed for `Pb_uniform_sample`. Set it to repeat a run exactly. |

`num_sites`, `num_augmentations` and `num_configurations` are the defaults of
`perovml.core.recipes.perov_adslab_generator` itself, so they are the same whichever entry point you
come in through. They do nothing for the `Pb_*` generators.

Typical orientation counts: 12–30 for `Pb_heuristic_sample` (the cone already removes most of the
space), 30–100 for `Pb_uniform_sample`. More orientations means better coverage and a proportionally
longer run; the samplers themselves are instant, the relaxations are not.

`Pb_heuristic_sample` walks a deterministic spiral over the cone and needs no seed. `Pb_uniform_sample`
draws random rotations, so it takes one: set `seed` to reproduce a candidate set, or leave it out and
the seed that was drawn is written to `provenance.json` in the run directory, which is enough to
regenerate the same run later. `perovml place` takes the same thing as `--seed` (§2).

### Calculator

| Key | Default | Meaning |
|---|---|---|
| `calculator` | **required** | `mock` (alias `MockCalculator`), `dpa3` (aliases `dpa3_omat24`, `dpa3omat24`), or `uma` (aliases `fairchem`, `fairchemcalculator`). Case-insensitive. |
| `dpa3_model_path` | `$DPA3_OMAT24_MODEL`, then `$DPA3_MODEL_PATH` | DPA-3 checkpoint. Raises if neither the key nor an environment variable is set. |
| `uma_model` | `$UMA_MODEL` | Pretrained name such as `uma-m-1p1`, or a checkpoint path. |
| `uma_task` | `oc20` | FAIRChem task head. |
| `uma_inference` | `default` | `default` or `turbo`. |
| `uma_device` | auto | `cuda`, `cpu`, or unset for automatic. |

An unrecognised name is an error that lists the accepted spellings. The mock calculator is used only
when named.

Which calculator, model file and task head actually ran is written to `provenance.json` in the run
directory (and to `metadata.json` under `calculator_provenance` for the batch scripts), including the
SHA-256 of a local checkpoint, so a number can always be traced back to what produced it.

### Relaxation

| Key | Default | Meaning |
|---|---|---|
| `optimizer_cls` | `LBFGS` | `LBFGS`, `BFGS` or `FIRE`. An unknown name is an error — silently substituting `LBFGS` would change the protocol without saying so. |
| `fmax` | `0.02` | Force convergence threshold, eV/Å. |
| `steps` | `100` | Maximum optimizer steps per structure. |

For a quick shakedown use `fmax: 0.5`, `steps: 10`; for production with an MLP, `fmax: 0.03–0.05`
and `steps: 120–200` is a reasonable band.

### Precomputed references (optional)

The adsorption energy needs a relaxed bare slab and a relaxed gas-phase molecule. Supplying either
skips that relaxation, which matters when you screen many molecules against one slab.

| Key | Meaning |
|---|---|
| `relaxed_slab_file` | Structure file of the already-relaxed slab. |
| `relaxed_slab_energy` | Its energy in eV. Supplying the energy is what actually skips the relaxation. |
| `relaxed_adsorbate_file` | Structure file of the already-relaxed gas-phase molecule. |
| `relaxed_adsorbate_energy` | Its energy in eV. |

### Output

`--out-root` (default `outputs`) receives a directory named `perovml_<YYYYmmdd>_<HHMMSS>`:

```
outputs/perovml_20260922_004820/
├── config.yaml              # snapshot of the config as parsed
├── binding_guess.txt        # only when binding_index was inferred
├── slab_relaxed.vasp        # relaxed bare slab reference
├── adsorbate_relaxed.vasp   # relaxed gas-phase molecule reference
├── initials/cand_N.vasp     # as generated
├── candidates/cand_N.vasp   # after relaxation
├── top.vasp                 # best candidate (lowest adsorption energy)
├── provenance.json          # calculator, model hash, optimizer, seed
├── report.csv
└── report.json
```

`report.csv` has four columns: `idx`, `energy` (eV), `adsorption_energy`
(E_adslab − E_slab − E_molecule, eV; negative means bound) and `anomalies`, a `;`-separated list
from FAIRChem's AdsorbML anomaly detection. Rows are sorted by adsorption energy where it is
available and by total energy otherwise, so `idx` is a rank, not a generation order.

---

## 2. `perovml place`

Generation only, no calculator, no relaxation. Useful for looking at what a sampler produces before
committing compute. Flags can also be supplied through `--config FILE` using the same names with
underscores.

```bash
perovml place \
    --slab-file data/slabs/FAPbI3_3x3_zcut017.vasp \
    --molecule-file data/molecules/Acetone.vasp \
    --mode Pb_heuristic_sample \
    --pb-num-orientations 8 \
    --out-root outputs/placements
```

| Flag | Config key | Default | Meaning |
|---|---|---|---|
| `--slab-file` | `slab_file` | — | One slab. Mutually exclusive with `--slab-dir`. |
| `--slab-dir` | `slab_dir` | — | Directory of slabs; each gets its own output subdirectory. |
| `--pattern` | `pattern` | `*.vasp` | Glob applied under `--slab-dir`. |
| `--molecule-file` | `molecule_file` | — | Adsorbate file. Mutually exclusive with `--adsorbate`. |
| `--adsorbate` | `adsorbate` | — | FAIRChem database key. |
| `--binding-index` | `binding_index` | guessed | As above. |
| `--which` | `which` | `single` | `single` for one adsorbate, `multiple` for co-adsorption. |
| `--mode` | `mode` | `random_site_heuristic_placement` | `random`, `heuristic`, `random_site_heuristic_placement`, `Pb_uniform_sample`, `Pb_heuristic_sample`. |
| `--num-sites` | `num_sites` | `50` | Sites, FAIRChem sampler only. |
| `--num-augmentations` | `num_augmentations` | `1` | Orientations per site, FAIRChem sampler only. |
| `--num-configurations` | `num_configurations` | `100` | Configurations, `--which multiple` only. |
| `--copies` | `copies` | `1` | Adsorbate copies, `--which multiple` only. |
| `--pb-num-orientations` | `pb_num_orientations` | `50` | Orientations for the `Pb_*` modes. |
| `--pb-cone-deg` | `pb_cone_deg` | `20.0` | Cone half-angle, `Pb_heuristic_sample` only. |
| `--interstitial-gap` | `interstitial_gap` | `0.1` | Extra clearance in Å above the covalent contact distance when lifting the molecule, `Pb_*` modes only. |
| `--surface-fraction` | `surface_fraction` | `0.25` | Fallback tagging: when the slab carries no tags *and* no Pb–I surface layer is found, mark this top fraction by *z* as surface. |
| `--min-ab` | `min_ab` | `8.0` | As above. |
| `--seed` | `seed` | `42` | Passed to the uniform sampler, and also seeds `random` and `numpy.random` for the FAIRChem samplers. |
| `--out-root` | `out_root` | `outputs` | Parent of the timestamped run directory. |

Note that `--mode` here selects the sampler, whereas in a `perovml run` config the sampler is
`generator` and `mode` means something narrower. This asymmetry is a wart; when in doubt, read the
`config.yaml` that both commands write into the run directory — it records what was actually used.

`--surface-fraction` is a fallback, used only when the Pb–I tagging finds no surface layer.
Tagging the slab yourself (tag `1` for surface, `0` for bulk) before it reaches `perovml` is always
better, and the log line

```
INFO perovml.utils.structure: Pb-I surface tagging: Pb:I = 9:18 (~0.50)
```

tells you what the code decided.

---

## 3. Batch workflow

`scripts/run_adsorption.py` (full MLP → optional DFT) and `scripts/run_parallel_simple.py`
(MLP only, lighter) take one slab and a *directory* of molecules. See
[adsorption_workflow.md](adsorption_workflow.md) for the task-directory layout and the resume rules.

### Common keys

| Key | Default | Meaning |
|---|---|---|
| `slab` | *required* | Slab file. Relaxed once and reused for every molecule. |
| `molecules_dir` | *required* | Directory of adsorbate structure files. |
| `pattern` | `*.vasp` | Glob applied under `molecules_dir`. |
| `calculator` | *required* (`run_parallel_simple.py`: `mock`) | `mock`, `dpa3` or `uma`, with the aliases in §1. An unrecognised name is an error. On the `run_parallel_simple.py` command line, `--calculator` accepts `mock` and `dpa3`. |
| `model_path` / `dpa3_model_path` | `$DPA3_OMAT24_MODEL`, `$DPA3_MODEL_PATH` | DPA-3 checkpoint. |
| `uma_model`, `uma_task`, `uma_device`, `uma_inference` | as in §1 | UMA settings, `scripts/run_adsorption.py` only. |
| `save_top` | `5` | Structures kept per molecule; `"all"` keeps every candidate. |
| `relaxed_slab_file`, `relaxed_slab_energy` | — | Skip the shared slab relaxation. |

### Output directory and resuming

`output` means something different in each script, and so does resuming. This is the one place
where reading the wrong row costs you a run.

| | `run_adsorption.py` | `run_parallel_simple.py` |
|---|---|---|
| `output` default | `outputs/adsorption` | `outputs/parallel_run` |
| What the run writes into | `<output>` exactly as given; nothing is appended | `<output>/<YYYYmmdd>_<HHMMSS>`, a fresh subdirectory on every invocation |
| How to resume | Point `output` at the existing directory. One that contains `metadata.json` is picked up where it stopped. | Pass `--resume <that timestamped subdirectory>`. Re-running with the same config does **not** resume; it starts a new run in a new subdirectory. |

`run_parallel_simple.py --resume` also inherits the stored configuration from the run's
`metadata.json`, so a resumed run keeps the calculator and model path it started with.

### Sampling and relaxation

The two scripts differ here, both in which keys they read and in what they default to.

| Key | `run_adsorption.py` | `run_parallel_simple.py` | Meaning |
|---|---|---|---|
| `fmax` | `0.03` | `0.05` | Force convergence threshold, eV/Å. |
| `steps` | `150` | `100` | Maximum optimizer steps. |
| `generator` | — | `Pb_heuristic_sample` | Single sampler: `Pb_uniform_sample`, `Pb_heuristic_sample`, `single`, `multi`. |
| `pb_num` | — | `24` | Orientations for that sampler. |
| `pb_num_uniform` | `30` | — | Orientations from `Pb_uniform_sample`. |
| `pb_num_heuristic` | `12` | — | Orientations from `Pb_heuristic_sample`. |
| `pb_cone` | `25.0` | `30.0` | Cone half-angle in degrees for the heuristic sampler. |

`run_adsorption.py` runs **both** samplers for every molecule, merges the two sets, and ranks the
relaxed candidates by anomaly severity first — no anomaly, then desorption only, then the more
serious anomalies — and by adsorption energy within one severity class. That ordering is
`perovml.core.recipes.select_best_by_priority`, and it is what `save_top` slices.

The per-molecule settings above are written into each task's `config.json` on its first run. On a
resume the saved values win: passing a different `fmax` or `steps` logs a mismatch warning and keeps
the original, so one task directory never mixes two convergence criteria. To change them, start a
new output directory.

### DFT stage (`scripts/run_adsorption.py`)

| Key | Default | Meaning |
|---|---|---|
| `run_dft` | `false` | Run the DFT stage after the MLP stage. |
| `dft_generate_only` | `false` | Write VASP inputs but do not execute VASP. |
| `dft_mode` | `scf` | `scf` for single points, `opt` for ionic relaxation. |
| `vasp_command` | `vasp_std` | Executable. Do not nest `srun` inside it when the job already runs under `srun`. |
| `potcar_dir` | `$VASP_PP_PATH` | POTCAR tree. |
| `dft_incar_override` | — | Mapping merged over the INCAR template. `DIPOL`, `NSW` and `IBRION` accept the literal string `auto`. |
| `slab_dft_dir` | — | Existing DFT directory for the bare slab; each task symlinks `dft/slab` to it instead of recomputing. |
| `dft_slab_energy`, `dft_molecule_energy` | — | Known reference energies in eV; supplying one skips that calculation. |
| `slab_mlp_relaxed` | — | The slab is already relaxed, so the MLP stage does a single point on it. |
| `disable_anomaly_detection` | `false` | Skip AdsorbML anomaly detection during ranking. |

INCAR defaults, POTCAR selection and Bader post-processing are documented in
[vasp_stage.md](vasp_stage.md).
