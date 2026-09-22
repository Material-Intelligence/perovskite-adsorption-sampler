# Model checkpoints

**No model weights are stored in this repository.** The machine-learned potential
used by `perovml` is published by a third party under its own terms; download it
yourself and point `perovml` at the file.

## DPA-3.1-3M (DeePMD-kit)

`perovml.calculators.DPA3Omat24Calculator` wraps DeePMD-kit's ASE calculator and
selects the multi-task head `Omat24`, so it consumes the multi-task checkpoint
`DPA-3.1-3M.pt` directly.

| | |
|---|---|
| Upstream | `deepmodelingcommunity/DPA-3.1-3M` on Hugging Face — <https://huggingface.co/deepmodelingcommunity/DPA-3.1-3M> |
| File | `DPA-3.1-3M.pt` |
| Licence | The model card states `cc-by-4.0`. The `Omat24` branch is trained on Meta's OMat24 dataset, itself released under CC-BY-4.0. Check both upstreams before redistributing anything derived from them. |
| Runtime | DeePMD-kit **v3.1.x** — the model card asks for v3.1.0, and the `dpa3` extra is pinned to `>=3.1,<3.2` so that `pip` cannot quietly give you a newer runtime. |

Download, for example with the Hugging Face CLI:

```bash
pip install -U "huggingface_hub[cli]"
hf download deepmodelingcommunity/DPA-3.1-3M DPA-3.1-3M.pt --local-dir models/dpa3
```

Then install the matching runtime and tell `perovml` where the checkpoint is:

```bash
pip install "perovml[dpa3]"                 # pulls deepmd-kit >=3.1,<3.2
export DPA3_MODEL_PATH=models/dpa3/DPA-3.1-3M.pt
```

The path is resolved in this order: the explicit setting (`--model-path` on the
command line, or `model_path` / `dpa3_model_path` in a YAML config), then
`$DPA3_OMAT24_MODEL`, then `$DPA3_MODEL_PATH`.

## Frozen single-branch models (LAMMPS only)

`perovml` does not need a frozen model. If you want one — for example to run the
`Omat24` branch inside LAMMPS — produce it locally from the checkpoint above with
DeePMD-kit, rather than looking for it here:

```bash
dp --pt freeze -c DPA-3.1-3M.pt -o DPA-3.1-3M-Omat24.pth --model-branch Omat24
```

Run `dp --pt show DPA-3.1-3M.pt` to list the branch names the checkpoint actually
carries; they must be spelled exactly.

## UMA / other FAIRChem models

`perovml run` and `perovml optimize` also accept a UMA calculator (`calculator:
uma`, or `--calc uma`), taking either a pretrained name such as `uma-m-1p1` or a
local checkpoint path via `uma_model` / `--model-path` / `$UMA_MODEL`. Those
checkpoints are FAIRChem's; see upstream
<https://github.com/facebookresearch/fairchem> for availability and terms.
Nothing is bundled here.
