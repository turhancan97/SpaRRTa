# SpaRRTa: A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models

Official implementation of **SpaRRTa**, a synthetic benchmark that probes whether Visual
Foundation Models (VFMs) — such as DINO, DINOv2/v3, MAE, CroCo, VGGT, SPA and CLIP — encode
the **spatial relations between objects** in a scene.

Unlike conventional 3D probing tasks that target metric quantities (depth, pose), SpaRRTa
measures a more fundamental capability: recognizing the *relative direction* (Front / Back /
Left / Right) of a target object with respect to a reference object, from a given viewpoint.

<p align="center">
  <img src="assets/teaser.png" alt="SpaRRTa teaser" width="100%">
</p>

> Turhan Can Kargın, Wojciech Jasiński, Adam Pardyl, Bartosz Zieliński, Marcin Przewięźlikowski.
> *SpaRRTa: A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models.*
> [arXiv:2601.11729](https://arxiv.org/abs/2601.11729)

---

## The task

SpaRRTa is a 4-way classification problem (**Front / Back / Left / Right**) with two variants:

- **SpaRRTa-ego (egocentric)** — directions are defined from the *camera's* viewpoint.
- **SpaRRTa-allo (allocentric)** — directions are defined from a *human figure's* viewpoint
  in the scene, which requires implicit perspective-taking.

The benchmark is rendered in Unreal Engine 5 across five environments (Forest, Desert,
Winter Town, Bridge, City), and is complemented by a small **real-world** set photographed with
lego figures for sim-to-real evaluation.

The overall data-generation and probing pipeline is summarized below:

<p align="center">
  <img src="assets/pipeline.png" alt="SpaRRTa data-generation and probing pipeline" width="100%">
</p>

## Probing strategies

A frozen backbone produces patch (and CLS) tokens; only a lightweight probe head is trained.
Three heads are provided (`sparrta/models/probes.py`):

| Head | Config target | Description |
|------|---------------|-------------|
| `ClassificationHead` | `sparrta.models.probes.ClassificationHead` | Linear probe on pooled features (GAP / CLS). |
| `ABMILPHead` | `sparrta.models.probes.ABMILPHead` | Attention-based multiple-instance pooling over patches. |
| `EfficientProbing` | `sparrta.models.probes.EfficientProbing` | Multi-query cross-attention aggregation (strongest). |

A central finding of the paper: spatial information lives in the **patch tokens**, so attention
probes (`EfficientProbing` > `ABMILPHead`) substantially outperform a linear probe on pooled features.

## Backbones

All 15 backbones from the paper are supported via Hydra configs in `configs/backbone/`:

| Config | Works out of the box | Notes |
|--------|:---:|-------|
| `dino_b16`, `dinov2_b14`, `dinov2_b14_reg`, `dinov2_l14_reg` | ✅ | `torch.hub` |
| `dinov3_timm` | ✅ | timm / Hugging Face |
| `mae_b16` | ✅ | Hugging Face `transformers` |
| `clip_b16_laion` | ✅ | `open_clip` |
| `deit3_b16` | ✅ | timm |
| `maskfeat_vitb16` | ⚠️ | needs `mmselfsup`/`mmcls` (see below) |
| `vggt_l16` | 🔧 | needs the external [VGGT](https://github.com/facebookresearch/vggt) repo → `$VGGT_REPO` |
| `spa_b16`, `spa_l16` | 🔧 | needs the external [SPA](https://github.com/HaoyiZhu/SPA) repo → `$SPA_REPO` |
| `croco_b16`, `crocov2_b16` | 🔧 | needs the external [CroCo](https://github.com/naver/croco) repo → `$CROCO_REPO` |
| `dinov3_b16` | 🔧 | needs the external [DINOv3](https://github.com/facebookresearch/dinov3) repo → `$DINOV3_REPO` and weights → `$DINOV3_WEIGHTS` |

Backbones marked 🔧 are loaded lazily: they only error (with a clear message) if you actually
select them without setting the corresponding environment variable. For the local DINOv3 you
can avoid all of this by using `backbone=dinov3_timm` instead.

## Installation

```bash
conda create -n sparrta python=3.9 --yes
conda activate sparrta
# Install PyTorch for your CUDA version (see https://pytorch.org), e.g.:
conda install pytorch=2.2.1 torchvision=0.17.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -e .
```

Optional, only for `maskfeat_vitb16`:

```bash
pip install -U openmim && mim install mmcv mmcls "mmselfsup>=1.0.0rc0"
```

## Data

The datasets are **not** bundled with the code. Download them from Hugging Face:

➡️ **https://huggingface.co/datasets/turhancan97/SpaRRTa**

Then point the code at the data via environment variables:

```bash
export SPARRTA_DATA_ROOT=/path/to/sparrta/unreal   # Unreal environments live here
export SPARRTA_LEGO_ROOT=/path/to/sparrta/lego      # real-world lego images
export SPARRTA_CACHE_DIR=./cache                    # cached frozen features (created on demand)
export SPARRTA_MODELS_DIR=~/.cache/sparrta/models   # downloaded backbone weights
```

### Expected on-disk layout

**Unreal (`$SPARRTA_DATA_ROOT`)** — one folder per environment, each holding image/annotation pairs:

```
$SPARRTA_DATA_ROOT/
  forest/mid-objects/
    img_0001.jpg
    params_0001.json
    ...
  desert/mid-objects/
  winter_town/mid-objects/
  bridge/mid-objects/
  city/mid-objects/
```

Each `params_*.json` stores the 3D positions used to compute the ground-truth direction:

```json
{
  "camera": { "location": { "x": 0.0, "y": 0.0 } },
  "actors": {
    "0": { "label": "Rock", "location": { "x": 1.0, "y": 2.0 } },
    "1": { "label": "Tree", "location": { "x": 3.0, "y": 4.0 } },
    "2": { "label": "Human", "location": { "x": 5.0, "y": 6.0 } }
  }
}
```

**Real-world lego (`$SPARRTA_LEGO_ROOT`)** — one folder per class:

```
$SPARRTA_LEGO_ROOT/
  front/  *.jpg
  back/   *.jpg
  left/   *.jpg
  right/  *.jpg
```

## Quick start

Train an `EfficientProbing` head on DINO features for the egocentric task in the forest environment:

```bash
python train.py \
  backbone=dino_b16 \
  dataset=unreal_position \
  probe=classifier probe._target_=sparrta.models.probes.EfficientProbing \
  dataset.perspective=camera \
  environment=forest
```

Switch the allocentric (human-perspective) task with `dataset.perspective=human`, and try other
backbones with `backbone=mae_b16`, `backbone=clip_b16_laion`, `backbone=dinov3_timm`, etc.
Results are appended to a CSV under `output_dir` (default `result/`).

Inspect a fully-resolved config without running anything:

```bash
python train.py --cfg job backbone=dino_b16 dataset=unreal_position
```

## Experiments & scripts

| Script | Purpose |
|--------|---------|
| `scripts/run_sweep.py` | Run a grid of backbones × layers (configured via `scripts/position_config.yaml` and env vars `PERSPECTIVE`, `ENVIRONMENT`, `PROBE_TYPE`). |
| `scripts/run_sweep_simple.py` | Minimal single-perspective launcher for quick tests. |
| `scripts/run_loto_fewshot.py` | Leave-one-environment-out transfer + few-shot adaptation matrix. |
| `scripts/run_lego_rebuttal.py` | Real-world (lego) sim-to-real evaluation matrix. |
| `scripts/summarize_loto_fewshot.py` | Aggregate LOTO/few-shot result CSVs into tables and plots. |
| `scripts/summarize_lego_rebuttal.py` | Aggregate real-world (lego) result CSVs. |
| `scripts/count_position_probe_params.py` | Report trainable parameter counts per probe head. |
| `scripts/slurm/*.sh` | SLURM batch wrappers (edit the conda activation lines for your cluster). |

### Transfer protocols

The training script supports three `protocol` values:

- `default` — standard train/val/test split within one environment.
- `loto_source_to_target` — train/val on non-holdout environments, test on the holdout one.
- `target_only` — train/val/test on the holdout environment (used for few-shot adaptation).

## Citation

```bibtex
@article{kargin2026sparrta,
  title   = {SpaRRTa: A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models},
  author  = {Karg{\i}n, Turhan Can and Jasi{\'n}ski, Wojciech and Pardyl, Adam and Zieli{\'n}ski, Bartosz and Przewi{\k{e}}{\'z}likowski, Marcin},
  journal = {arXiv preprint arXiv:2601.11729},
  year    = {2026}
}
```

## License

Released under the [MIT License](LICENSE).
