# Notebooks

## `position_probe_inspection.ipynb`

Single-image inference + attention visualization for a trained **position-between-objects**
probe. It loads a frozen backbone and a trained probe head, predicts the relative direction
(Front / Back / Left / Right) of a target object w.r.t. a reference object, and overlays the
probe's attention for the attention-based heads.

### Bundled assets (committed)

- `examples/` — four `winter_town_2` scenes (one per direction) with their `params_*.json`
  so the notebook can show ground-truth labels.
- `heads/head_vggt_l16_cls_efficient_winter_town_2_camera_Snowman_Husky.pt` — a trained
  `vggt_l16` / `cls_efficient` head (Snowman → Husky, camera view).

### Run it

```bash
pip install -e .                 # from the repo root
export VGGT_REPO=/path/to/vggt   # this example uses the VGGT backbone
cd notebook
jupyter lab position_probe_inspection.ipynb
```

The notebook resolves `examples/` and `heads/` relative to this directory; override them with
`SPARRTA_NOTEBOOK_EXAMPLES` / `SPARRTA_HEADS_DIR` if you run it from elsewhere. To inspect your
own runs, set `SPARRTA_HEADS_DIR` to the directory where training saved your `*.pt` heads (use
`save_head=true`) and adjust the parameters in the second cell.
