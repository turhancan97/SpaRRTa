#!/usr/bin/env python3
"""Create and upload the SpaRRTa attention-analysis split to a Hugging Face dataset repo.

This is the image + segmentation-mask asset used by the attention-analysis suite
(`sparrta/analysis/`). Unlike the main SpaRRTa probing data, every scene ships with
per-object masks, so it is stored as a plain folder tree (NOT ImageFolder):

    <images-dir>/<environment>/params_XXXX/img_XXXX.jpg
    <images-dir>/<environment>/params_XXXX/metadata/mask_Human.png
    <images-dir>/<environment>/params_XXXX/metadata/mask_Tree.png
    <images-dir>/<environment>/params_XXXX/metadata/mask_Truck.png
    <images-dir>/<environment>/params_XXXX/metadata/masks_log.csv

`upload_folder` preserves this layout, so:
    huggingface-cli download <repo-id> --repo-type dataset --local-dir X
reproduces `X/<environment>/params_XXXX/...`, which the analysis code reads when you
point SPARRTA_ANALYSIS_ROOT at X.

Usage:
    export HF_TOKEN=...   # a write token for the turhancan97 namespace
    # source images live in the (private) midvision-probe repo: analysis/images
    python upload_attention_to_hf.py \
        --images-dir /path/to/midvision-probe/analysis/images
    # add --dry-run to scan/build the card without uploading
"""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
# Upload images, masks and the per-scene mask logs; never OS metadata or caches.
ALLOW_PATTERNS = ["*.jpg", "*.jpeg", "*.png", "*.bmp", "*.webp", "*.tif", "*.tiff", "*.csv"]
ALLOW_PATTERNS += [p.upper() for p in ALLOW_PATTERNS]
IGNORE_PATTERNS = ["**/.DS_Store", ".DS_Store", "**/Thumbs.db", "**/__pycache__/**"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Upload the SpaRRTa attention-analysis split to HF.")
    p.add_argument("--repo-id", default="turhancan97/SpaRRTa-Attention",
                   help="Hugging Face dataset repo id (default: turhancan97/SpaRRTa-Attention)")
    p.add_argument("--images-dir", type=Path,
                   default=Path(os.environ.get("SPARRTA_ANALYSIS_ROOT", "./data/attention")),
                   help="Directory containing per-environment scene folders "
                        "(defaults to $SPARRTA_ANALYSIS_ROOT or ./data/attention)")
    p.add_argument("--token", default=os.getenv("HF_TOKEN"),
                   help="HF token (defaults to HF_TOKEN env var)")
    p.add_argument("--private", action="store_true", default=False,
                   help="Create a private repo (default: public / open access)")
    p.add_argument("--license", default="mit", help="License string for the card (default: mit)")
    p.add_argument("--dry-run", action="store_true", help="Scan + build card only; no upload")
    p.add_argument("--commit-message", default=None, help="Optional custom commit message")
    return p.parse_args()


def scan_environments(images_dir: Path) -> Dict[str, int]:
    """Return {environment: scene_count}. A scene is a params_* folder holding an image."""
    counts: Dict[str, int] = {}
    for env_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        scene_count = 0
        for scene in env_dir.iterdir():
            if not scene.is_dir():
                continue
            if any(f.suffix.lower() in IMAGE_EXTENSIONS for f in scene.iterdir() if f.is_file()):
                scene_count += 1
        if scene_count:
            counts[env_dir.name] = scene_count
    if not counts:
        raise FileNotFoundError(
            f"No environment/scene folders with images found under {images_dir}"
        )
    return counts


def build_readme(repo_id: str, license_name: str, counts: Dict[str, int]) -> str:
    total = sum(counts.values())
    generated = datetime.now(timezone.utc).isoformat()
    rows = "\n".join(f"| `{env}` | {counts[env]:,} |" for env in counts)
    return f"""---
license: {license_name}
task_categories:
- image-feature-extraction
language:
- en
size_categories:
- n<1K
pretty_name: SpaRRTa-Attention
tags:
- spatial-reasoning
- spatial_intelligence
- vision-foundation-models
- probing
- interpretability
- attention
- segmentation-masks
- arxiv:2601.11729
---

<h1 style="display:flex; align-items:center; gap:10px;">
  <span style="color:#FF7096;">SpaRRTa-Attention</span>: Attention-Analysis Split of the SpaRRTa Benchmark
</h1>

**SpaRRTa-Attention** is the interpretability asset for the synthetic
[**SpaRRTa**](https://huggingface.co/datasets/turhancan97/SpaRRTa) benchmark. Each scene ships
with **per-object segmentation masks** so that a frozen Visual Foundation Model's self-attention
can be measured *between the objects in the scene* (Human / Tree / Truck), the CLS token,
the background, and register tokens.

- 📄 **Paper:** [arXiv:2601.11729](https://arxiv.org/abs/2601.11729)
- 💻 **Code:** [github.com/gmum/SpaRRTa](https://github.com/gmum/SpaRRTa) (see `sparrta/analysis/`)
- 🧩 **Main (synthetic) split:** [turhancan97/SpaRRTa](https://huggingface.co/datasets/turhancan97/SpaRRTa)

## Contents

- **Total scenes:** **{total:,}** across {len(counts)} environments.
- One folder per environment; one folder per scene, each with the rendered image and its masks:

| environment | scenes |
|---|---:|
{rows}

```
<environment>/params_XXXX/
  img_XXXX.jpg
  metadata/
    mask_Human.png
    mask_Tree.png
    mask_Truck.png
    masks_log.csv
```

Masks are binary PNGs aligned to the image; `masks_log.csv` records the per-object mask metadata.

Generated on {generated}.

## Use with the SpaRRTa code

Download the dataset, then point the analysis code at it:

```bash
huggingface-cli download {repo_id} --repo-type dataset --local-dir ./hf_SpaRRTa-Attention
export SPARRTA_ANALYSIS_ROOT=$(pwd)/hf_SpaRRTa-Attention
```

Then run the attention analysis (see the [code repository](https://github.com/gmum/SpaRRTa)):

```bash
python sparrta/analysis/compute_attention.py environment=winter_town
```

## License

Released under the [MIT License](https://opensource.org/license/mit).

## Citation

```bibtex
@misc{{kargin2026sparrta,
  title={{SpaRRTa: A Synthetic Benchmark for Evaluating Spatial Intelligence in Visual Foundation Models}},
  author={{Turhan Can Kargin and Wojciech Jasiński and Adam Pardyl and Bartosz Zieliński and Marcin Przewięźlikowski}},
  year={{2026}},
  eprint={{2601.11729}},
  archivePrefix={{arXiv}},
  primaryClass={{cs.CV}},
  url={{https://arxiv.org/abs/2601.11729}}
}}
```
"""


def main() -> None:
    args = parse_args()
    images_dir = args.images_dir.resolve()
    if not images_dir.is_dir():
        raise FileNotFoundError(f"images dir does not exist: {images_dir}")

    counts = scan_environments(images_dir)
    total = sum(counts.values())
    print(f"Scanning {images_dir}")
    print("Scenes per environment:")
    for env, n in counts.items():
        print(f"  {env}: {n}")
    print(f"  total: {total}")

    readme = build_readme(args.repo_id, args.license, counts)

    if args.dry_run:
        print("\n--- DRY RUN: dataset card preview ---\n")
        print(readme)
        print("\nNo repo created, no files uploaded.")
        return

    if not args.token:
        print("ERROR: no HF token. Set HF_TOKEN or pass --token.", file=sys.stderr)
        sys.exit(1)

    from huggingface_hub import HfApi

    api = HfApi(token=args.token)
    print(f"\nEnsuring dataset repo exists (public={not args.private}): {args.repo_id}")
    api.create_repo(
        repo_id=args.repo_id,
        repo_type="dataset",
        private=args.private,
        exist_ok=True,
    )

    # 1) Upload the README/dataset card.
    with tempfile.TemporaryDirectory(prefix="attention_card_") as tmp:
        card_path = Path(tmp) / "README.md"
        card_path.write_text(readme, encoding="utf-8")
        api.upload_file(
            path_or_fileobj=str(card_path),
            path_in_repo="README.md",
            repo_id=args.repo_id,
            repo_type="dataset",
            commit_message="Add/update dataset card",
        )
    print("Uploaded README.md")

    # 2) Upload the scene tree (images + masks + logs), preserving <env>/params_XXXX/ layout.
    commit_message = args.commit_message or (
        f"Add {total} attention-analysis scenes "
        f"({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')})"
    )
    print(f"Uploading {total} scenes from {images_dir} ...")
    api.upload_folder(
        folder_path=str(images_dir),
        repo_id=args.repo_id,
        repo_type="dataset",
        allow_patterns=ALLOW_PATTERNS,
        ignore_patterns=IGNORE_PATTERNS,
        commit_message=commit_message,
    )

    print("\nUpload complete.")
    print(f"Dataset: https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
