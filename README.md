# perovskite-adsorption-sampler

Two adsorbate-placement samplers for Pb–I terminated perovskite surfaces, built on top of
[FAIRChem](https://github.com/facebookresearch/fairchem), with a resumable machine-learned-potential
(MLP) to DFT screening workflow. The Python package and the console script are both named `perovml`.

This is a domain-specific layer, not a simulation engine. `PbAdsorbateSlabConfig` subclasses
FAIRChem's `AdsorbateSlabConfig`; energies and forces come from whatever ASE calculator you plug in;
DFT is delegated to VASP. What the package contributes is the placement logic for a Pb–I surface,
the bookkeeping that makes a long screening run restartable, and a VASP input/parsing stage that
follows on from the MLP stage.

## What it does

- **Two samplers for a Pb site.** Both pick the surface Pb atom nearest the cell centre, using a
  minimum-image (torus) metric in fractional *xy*, and then differ in how they orient the molecule:
  - `Pb_uniform_sample` — uniform SO(3) rotations from Marsaglia quaternions, molecule anchored at
    its centre of mass. Use it when you do not know how the molecule binds.
  - `Pb_heuristic_sample` — orientations restricted to a cone of half-angle θ about the surface
    normal, sampled on a structured (β, ψ, φ) grid, molecule anchored at its *binding atom*. Use it
    when you do.

  Both lift the molecule along the surface normal by a computed interstitial gap and tag the merged
  adslab (bulk `0`, surface `1`, adsorbate `2`) so downstream FAIRChem code treats it correctly.
  See [docs/pb_sampling_algorithm.md](docs/pb_sampling_algorithm.md).
- **A resumable MLP→DFT workflow.** One self-contained task directory per molecule, holding
  `config.json` and `status.json`. Re-running the same command picks up at the stage each molecule
  actually reached, so a wall-clock kill costs you the configurations in flight, not the run.
  See [docs/adsorption_workflow.md](docs/adsorption_workflow.md).
- **A VASP stage.** INCAR templates for single-point and relaxation (ENCUT 500, `IVDW = 12` for
  DFT-D3(BJ), `LAECHG` for Bader), POTCAR concatenation with an element-priority table, automatic
  `DIPOL` from the mean fractional coordinate, and Bader charge-transfer post-processing.
  See [docs/vasp_stage.md](docs/vasp_stage.md).
- **Data-parallel sharding for Slurm.** `numpy.array_split` over `SLURM_PROCID` / `SLURM_NTASKS`,
  one result file per rank, plus a reducer and a completeness check. No MPI communicator is needed.
  See [docs/architecture.md](docs/architecture.md).
- **A mock calculator.** `perovml.calculators.MockCalculator` returns meaningless but well-formed
  energies and forces, so the whole pipeline can be exercised on a laptop, with no GPU and no model
  download, before you spend allocation on it.

## Install

Python ≥ 3.11 (fairchem-core 2.x sets the floor). The heavy dependency is FAIRChem, which pulls in
PyTorch.

```bash
git clone https://github.com/xiejiahao/perovskite-adsorption-sampler.git
cd perovskite-adsorption-sampler

uv venv --python 3.11
source .venv/bin/activate
uv pip install -e ".[dev]"
```

This is a plain setuptools package with a `src` layout, so `python -m pip install -e ".[dev]"` into
a virtual environment does the same job if you would rather not use `uv`. Four extras are optional
(`perovml[all]` is `dpa3` + `smiles` + `mpi`):

| Extra | Pulls in | Needed for |
|---|---|---|
| `dpa3` | `deepmd-kit` | the DPA-3 calculator |
| `smiles` | `rdkit` | `scripts/gen_from_smiles.py`, which builds molecules from SMILES |
| `mpi` | `mpi4py` | rank discovery in `scripts/optimize_molecules.py`; it falls back to the Slurm environment variables without it |
| `heuristic` | nothing new | Compatibility alias. pymatgen is a core dependency, so this extra installs nothing that a plain install does not already give you |

A CPU-only PyTorch is enough for everything in this README; the samplers themselves do no tensor
work, and only a real MLP calculator wants a GPU. If the default wheel is larger than you need,
install the CPU build from [pytorch.org](https://pytorch.org/get-started/locally/) first.

## A 60-second example

`examples/01_place_and_screen.py` loads the shipped FAPbI₃ slab and an acetone molecule, generates
configurations with both samplers, relaxes them with `MockCalculator`, prints a ranked table and
writes the best structure to `example_output/`. It needs no GPU and downloads nothing.

```bash
python examples/01_place_and_screen.py
```

```
slab     : FAPbI3_3x3_zcut017.vasp (H5PbCI3N2)
molecule : Acetone.vasp (H6C3O)
sampling : 10 orientations x 2 samplers, mock calculator

rank  config          sampler                 E_ads (eV)     dE (eV)
--------------------------------------------------------------------
   1  uniform_003     Pb_uniform_sample         -67.0824      0.0000
   2  uniform_001     Pb_uniform_sample         -67.0589      0.0235
   3  uniform_008     Pb_uniform_sample         -66.8556      0.2269
...
best configuration written to example_output/best_adslab.vasp
total wall clock: 0.3 s
```

Those energies come from the mock calculator and carry no physical meaning; the script says so too.
The point is that the plumbing runs end to end before you commit a GPU to it.

The same thing from the command line, in two steps. First, placements only — no energies:

```bash
perovml place \
    --slab-file data/slabs/FAPbI3_3x3_zcut017.vasp \
    --molecule-file data/molecules/Acetone.vasp \
    --mode Pb_heuristic_sample \
    --pb-num-orientations 8 \
    --out-root outputs/placements
```

```
INFO perovml.utils.structure: guessed binding index = 3 (Z=8)
INFO perovml.utils.structure: Pb-I surface tagging: Pb:I = 9:18 (~0.50)
[Info] Run directory: outputs/placements/20260922_010424
[Done] FAPbI3_3x3_zcut017: wrote 8 placements to outputs/placements/20260922_010424/FAPbI3_3x3_zcut017
```

Those two `INFO` lines are the ones worth reading: which atom the sampler will point at the surface,
and whether the slab was recognised as Pb–I terminated with a sensible ratio.

Then place *and* relax, driven by a config file. Write this to `my_run.yaml`:

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

The run directory holds a snapshot of the config, the relaxed slab and gas-phase references, every
generated structure under `initials/`, every relaxed structure under `candidates/`, the best one as
`top.vasp`, a `provenance.json` recording which calculator, model and seed produced the numbers, and
a ranked `report.csv` / `report.json`:

```csv
idx,energy,adsorption_energy,anomalies
0,476.7260481800209,-80.69844607478844,
1,476.72672620097586,-80.69776805383346,
2,478.69855240445133,-78.72594185035798,
```

Those numbers come from the mock calculator and mean nothing physically. Swap in a real calculator
(`calculator: dpa3`, or `uma`) and the same commands produce numbers that do.

## Command line

```
perovml [--version] {run,place,optimize,sd} ...
```

| Subcommand | What it does |
|---|---|
| `perovml run -c CONFIG` | Sample placements and relax them, driven by a YAML config |
| `perovml place --slab-file ... --molecule-file ...` | Generate placements only, no energy evaluation |
| `perovml optimize INPUT -o OUTPUT` | Relax one structure with an MLP calculator |
| `perovml sd INPUT --z-cut 0.26` | Write VASP selective-dynamics flags onto a POSCAR |

`perovml sd` is the quickest one to try, because it touches neither FAIRChem nor a calculator:

```bash
perovml sd data/slabs/FAPbI3_3x3.vasp -o slab_sd.vasp --z-cut 0.26
```

```
Wrote slab_sd.vasp
  sites: 324  fully or partly relaxed: 108
```

Batch screening over a directory of molecules is driven by the scripts in `scripts/` rather than by
the console script; see [docs/adsorption_workflow.md](docs/adsorption_workflow.md).

## Configuration

Every key of every config schema — the `perovml run` config, the `perovml place` options, and the
batch-workflow config that `scripts/run_adsorption.py` reads — is listed with its default and its
effect in [docs/parameters.md](docs/parameters.md). Ready-made configs live in `examples/configs/`:

| File | Consumed by | Purpose |
|---|---|---|
| `examples/configs/mock.yaml` | `scripts/run_parallel_simple.py` | Smoke test: one molecule, four orientations, no model |
| `examples/configs/parallel.yaml` | `scripts/run_parallel_simple.py` | Sharded sampling over a molecule directory |
| `examples/configs/dpa3.yaml` | `scripts/run_adsorption.py` | Full MLP screening with optional VASP verification |

All paths in them are relative to the repository root, and no cluster-specific value is baked in.

## Models

**No model weights ship with this repository.** The MLP checkpoints are third-party and carry their
own terms. [models/README.md](models/README.md) gives the upstream source, the download command, the
licence note for each, and the environment variables (`$DPA3_MODEL_PATH`, `$UMA_MODEL`) that
`perovml` reads so that no path has to be hard-coded.

## Running on a cluster

`examples/slurm/optimize_molecules.sbatch` is a submission template with `<PLACEHOLDER>` fields for
the account, partition and environment — directive names differ between sites, so check yours.
[examples/03_slurm_parallel.md](examples/03_slurm_parallel.md) explains the sharding model, the
relationship between `srun -n` and the number of shards, and how to verify that a parallel run
covered every task. [examples/02_dpa3_screening.md](examples/02_dpa3_screening.md) is a two-mode
walkthrough: run the same molecule under both samplers and compare what they find.

## Repository layout

```
src/perovml/
    core/          placement.py (the two samplers), recipes.py (generate → relax → rank)
    recipes/       adsorption.py (the resumable per-molecule task)
    dft/           vasp.py (input generation, execution, parsing, Bader)
    parallel/      shard.py, reduce.py (Slurm sharding and result merging)
    calculators/   mock.py, dpa3.py
    utils/         structure.py, poscar_tools.py
    cli/           main.py and one module per subcommand
scripts/           batch runners and structure-preparation tools
examples/          runnable example, configs, Slurm template, walkthroughs
docs/              algorithm, architecture, workflow, VASP stage, parameter reference
data/              two FAPbI₃ slabs and ten probe molecules
tests/             pytest suite
```

## Documentation

| Document | Contents |
|---|---|
| [docs/pb_sampling_algorithm.md](docs/pb_sampling_algorithm.md) | The two samplers in detail: site selection, rotation sampling, anchoring, lifting |
| [docs/adsorption_workflow.md](docs/adsorption_workflow.md) | The MLP→DFT workflow, its task-directory layout and its resume semantics |
| [docs/architecture.md](docs/architecture.md) | Package layout, runtime data flow, and the Slurm sharding model |
| [docs/vasp_stage.md](docs/vasp_stage.md) | INCAR defaults, POTCAR assembly, `DIPOL`, Bader analysis |
| [docs/parameters.md](docs/parameters.md) | Every configuration key, with defaults and typical values |

## Citing

`CITATION.cff` is in the repository root; GitHub renders a "Cite this repository" button from it.

## Licence

MIT — see [LICENSE](LICENSE). Third-party components and what is derived from them are listed in
[NOTICE](NOTICE). Model checkpoints are not covered by this licence and are not distributed here.

## Acknowledgements

This package is built on [FAIRChem](https://github.com/facebookresearch/fairchem) (its
`AdsorbateSlabConfig` and its AdsorbML anomaly detection),
[ASE](https://gitlab.com/ase/ase) (structures, optimizers, calculator interface, I/O),
[pymatgen](https://github.com/materialsproject/pymatgen) (structure and VASP file handling), and
optionally [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit) (the DPA-3 calculator).
