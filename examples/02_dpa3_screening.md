# Screening with a machine-learned potential

`examples/01_place_and_screen.py` shows the mechanics with a mock calculator. This walkthrough is
the real thing: the same molecule screened under **both** samplers with a DPA-3 potential, the two
sets compared, and then the same recipe scaled to a directory of molecules.

Nothing here is specific to a cluster. Paths are relative to the repository root, and the one
machine-specific value — where your model checkpoint lives — is an environment variable.

## Before you start

```bash
uv pip install -e ".[dpa3]"
```

Download a DPA-3 checkpoint as described in [models/README.md](../models/README.md), then:

```bash
export DPA3_MODEL_PATH=<PATH_TO_YOUR_CHECKPOINT>/DPA-3.1-3M.pt
```

Every command below reads that variable, so no path is written into a config file.

## Step 0: rehearse with the mock calculator

Do this first, every time you change a config. It takes seconds, needs no GPU, and catches the
mistakes that are expensive to discover three hours into a job: a wrong path, a molecule whose
binding atom was guessed badly, a slab with no surface tags.

Write `heuristic.yaml`:

```yaml
slab_file: data/slabs/FAPbI3_3x3_zcut017.vasp
molecule_file: data/molecules/DMSO.vasp
generator: Pb_heuristic_sample
pb_num: 12
pb_cone: 30.0
min_ab: 8.0
calculator: MockCalculator
optimizer_cls: LBFGS
fmax: 0.5
steps: 10
```

and `uniform.yaml`, which differs in three lines — a different sampler, more orientations, and no
cone, because the uniform sampler does not use one:

```yaml
generator: Pb_uniform_sample
pb_num: 24
# no pb_cone
```

```bash
perovml run -c heuristic.yaml --out-root outputs/compare
perovml run -c uniform.yaml   --out-root outputs/compare
```

```
[Done] perovml run wrote 12 candidates to outputs/compare/perovml_20260922_013654
[Done] perovml run wrote 24 candidates to outputs/compare/perovml_20260922_013659
```

Check the log for the two lines that tell you whether the setup makes sense:

```
INFO perovml.utils.structure: Pb-I surface tagging: Pb:I = 9:18 (~0.50)
INFO perovml.utils.structure: guessed binding index = 3 (Z=8)
```

The first says the surface was identified as Pb–I terminated with a sensible ratio; a wildly
different ratio means the tagging is wrong and the Pb site selection cannot be trusted. The second
says which atom the heuristic sampler will point at the surface — for DMSO that is the oxygen,
which is right. If it guesses wrong, set `binding_index` explicitly.

## Step 1: switch to DPA-3

One key changes:

```yaml
calculator: dpa3
```

Leave `dpa3_model_path` out and the calculator reads `$DPA3_OMAT24_MODEL`, then `$DPA3_MODEL_PATH`.
Tighten the relaxation at the same time — the mock settings above are deliberately sloppy:

```yaml
fmax: 0.03
steps: 150
```

```bash
perovml run -c heuristic.yaml --out-root outputs/compare
perovml run -c uniform.yaml   --out-root outputs/compare
```

Expect minutes per configuration on a GPU for a molecule of this size, so a 24-orientation uniform
sweep is a small job, not an interactive one. Run it under your scheduler; see
[03_slurm_parallel.md](03_slurm_parallel.md).

## Step 2: compare the two runs

Each run directory holds a `report.csv` with `idx`, `energy`, `adsorption_energy` and `anomalies`,
already sorted best-first.

```python
import glob
import pandas as pd

frames = []
for run_dir in sorted(glob.glob("outputs/compare/perovml_*")):
    df = pd.read_csv(f"{run_dir}/report.csv")
    df["run"] = run_dir
    frames.append(df)

all_runs = pd.concat(frames, ignore_index=True)
print(all_runs.groupby("run")["adsorption_energy"].agg(["count", "min", "mean", "std"]).round(3))
```

```
                                         count     min    mean    std
run
outputs/compare/perovml_20260922_013654     12  25.612  29.308  1.872
outputs/compare/perovml_20260922_013659     24  26.113  27.954  1.029
```

(Those are mock-calculator numbers from Step 0, which is why they are positive and meaningless. With
DPA-3 the `min` column is the one that matters, and it should be negative for a molecule that binds.
The uniform row also moves from run to run: `perovml run` takes no random seed, so
`Pb_uniform_sample` draws fresh rotations each time, while the cone sampler walks a structured grid
and reproduces its row unchanged on every re-run. Only `perovml place` accepts `--seed`.)

What to look at:

- **`min`** — the best configuration each sampler found. If the cone sampler's best is close to the
  uniform sampler's best, the cone was aimed correctly and you can use the cheaper sampler from then
  on. If the uniform sampler found something substantially lower, your assumption about how the
  molecule binds was wrong.
- **`std`** — how spread out the candidates are. A cone sweep with a large spread usually means the
  cone is too wide, or the molecule is flexible enough that orientation is not the whole story.
- **`anomalies`** — a non-empty entry means the AdsorbML detector thinks the relaxation ended
  somewhere other than a bound adsorbate: desorbed, dissociated, or the slab itself moved.
  Anomalous candidates are ranked last regardless of energy, and an anomaly on the *best* energy is
  a result that needs looking at, not a number to report.

Inspect the winner:

```bash
ase gui outputs/compare/perovml_20260922_013654/top.vasp
```

## Step 3: scale up to many molecules

For more than a handful of molecules, stop writing one config per run and use the batch workflow,
which does both samplers in a single pass per molecule, reuses one relaxed slab across all of them,
and is resumable:

```bash
python scripts/run_adsorption.py --config examples/configs/dpa3.yaml
```

`examples/configs/dpa3.yaml` is the same idea in batch form: a `molecules_dir` instead of one
`molecule_file`, `pb_num_uniform` and `pb_num_heuristic` instead of one `generator`, and an optional
VASP stage on the survivors. [docs/adsorption_workflow.md](../docs/adsorption_workflow.md) covers the
task-directory layout and the resume rules; [docs/parameters.md](../docs/parameters.md) lists every
key.

## Choosing between the samplers

| | `Pb_heuristic_sample` | `Pb_uniform_sample` |
|---|---|---|
| Anchor | the binding atom | the centre of mass |
| Orientations | inside a cone of half-angle `pb_cone` about the surface normal | all of SO(3) |
| Typical count | 12–30 | 30–100 |
| Assumes | you know which atom binds, and roughly how it points | nothing |
| Fails when | the assumption is wrong — quietly, by returning a plausible but non-optimal structure | never, but it needs many more samples to cover the space |

The useful pattern is the one in this walkthrough: run both once on a representative molecule,
confirm they agree, and then use the cone sampler for the rest of the series. Running only the cone
sampler from the start, without that check, is the version of this that goes wrong.

## If something does not work

| Symptom | Likely cause |
|---|---|
| The run finishes instantly and the energies look arbitrary | `calculator` was misspelled. Anything unrecognised falls back to the mock calculator; check the first lines of the log. |
| `ValueError` about a missing model path | Neither `dpa3_model_path` nor `$DPA3_MODEL_PATH` / `$DPA3_OMAT24_MODEL` is set. |
| Every candidate is flagged as an anomaly | Usually the slab: wrong surface tags, or a vacuum gap too small for the molecule. Run `perovml place` and look at the generated structures before spending compute on them. |
| The binding index is guessed wrong | Set `binding_index` explicitly; it is 0-based over the molecule file's own atom order. |
| A warning about small `a`/`b` lengths | The slab is smaller than `min_ab`. The molecule may interact with its own periodic image; enlarge the supercell rather than lowering the threshold. |
