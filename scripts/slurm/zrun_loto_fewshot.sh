#!/bin/bash
# ----------------------------------------------------------------------------
# SLURM directives -- EDIT FOR YOUR CLUSTER. Set the partition/QOS to the names
# your site uses (run `sinfo` to list them); tune the resource requests below.
# ----------------------------------------------------------------------------
#SBATCH -p <your_partition>
#SBATCH --qos <your_qos>
#SBATCH --gpus=1
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=unreal_loto
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

#=============================================================================
# CONFIGURATION - Edit these variables if needed
#=============================================================================
OUTPUT_DIR="result_unreal_loto_rebuttal"
CONDA_ENV="${CONDA_ENV:-sparrta}"  # name of your conda env
DRY_RUN=false         # true/false
SKIP_EXISTING=true    # true/false
SKIP_SUMMARY=false    # true/false
BACKBONES="vggt_l16" # dino_b16,dinov2_b14,dinov2_b14_reg,dinov2_l14_reg,dinov3_b16,dinov3_timm,croco_b16,crocov2_b16,mae_b16,maskfeat_vitb16,spa_b16,spa_l16,vggt_l16,deit3_b16,clip_b16_laion
HEADS="EfficientProbing" # EfficientProbing,GAP
PERSPECTIVES="human" # camera,human
HOLDOUT_FOLDS="winter_town_2" # bridge_2,city_2,desert_2,forest_2,winter_town_2
PAIR_SWAP_MODE=true # true/false
PAIR_SWAP_SOURCE_ENV="desert_2"
PAIR_SWAP_TARGET_ENV="desert_3"
LEGO_HOLDOUT_MODE=false # true/false
LEGO_SPLIT_MODE="time_series" # random,time_series,clip_block_random,clip_block_time_series
LEGO_DATASET_ROOT="${SPARRTA_LEGO_ROOT:-./data/lego_images}"
RESULT_CSV_SUFFIX="" # optional; if empty and LEGO_HOLDOUT_MODE=true, launcher auto-uses lego_holdout

#=============================================================================
# SETUP
#=============================================================================
# Point the code at the data. Set these to where you downloaded the SpaRRTa
# datasets (see the README "Data" section). Override them by exporting in your
# shell or via `sbatch --export`; the defaults below assume a local ./data dir.
export SPARRTA_DATA_ROOT="${SPARRTA_DATA_ROOT:-./data}"
export SPARRTA_LEGO_ROOT="${SPARRTA_LEGO_ROOT:-./data/lego_images}"
export SPARRTA_CACHE_DIR="${SPARRTA_CACHE_DIR:-./cache}"
export HYDRA_FULL_ERROR=1

nvidia-smi -L

conda init bash
# Activate your environment (edit path to your conda install)
source "${CONDA_PREFIX_PATH:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

mkdir -p logs

if [ "${PAIR_SWAP_MODE}" = "true" ] && [ "${LEGO_HOLDOUT_MODE}" = "true" ]; then
  echo "Error: PAIR_SWAP_MODE and LEGO_HOLDOUT_MODE cannot both be true."
  exit 1
fi

CMD=(
  python scripts/run_loto_fewshot.py
  --output-dir "${OUTPUT_DIR}"
  --backbones "${BACKBONES}"
  --heads "${HEADS}"
  --perspectives "${PERSPECTIVES}"
  --holdout-folds "${HOLDOUT_FOLDS}"
)

if [ "${DRY_RUN}" = "true" ]; then
  CMD+=(--dry-run)
fi
if [ "${SKIP_EXISTING}" = "true" ]; then
  CMD+=(--skip-existing)
fi
if [ "${SKIP_SUMMARY}" = "true" ]; then
  CMD+=(--skip-summary)
fi
if [ "${PAIR_SWAP_MODE}" = "true" ]; then
  CMD+=(--pair-swap-mode)
  CMD+=(--pair-swap-source-env "${PAIR_SWAP_SOURCE_ENV}")
  CMD+=(--pair-swap-target-env "${PAIR_SWAP_TARGET_ENV}")
fi
if [ "${LEGO_HOLDOUT_MODE}" = "true" ]; then
  CMD+=(--real-world-holdout-mode)
  CMD+=(--real-world-split-mode "${LEGO_SPLIT_MODE}")
  CMD+=(--real-world-dataset-root "${LEGO_DATASET_ROOT}")
fi
if [ -n "${RESULT_CSV_SUFFIX}" ]; then
  CMD+=(--result-csv-suffix "${RESULT_CSV_SUFFIX}")
fi

echo "=============================================="
echo "SLURM Job Configuration"
echo "=============================================="
echo "Job ID:         ${SLURM_JOB_ID}"
echo "Job Name:       ${SLURM_JOB_NAME}"
echo "Output dir:     ${OUTPUT_DIR}"
echo "Conda env:      ${CONDA_ENV}"
echo "Backbones:      ${BACKBONES}"
echo "Heads:          ${HEADS}"
echo "Perspectives:   ${PERSPECTIVES}"
echo "Holdout folds:  ${HOLDOUT_FOLDS}"
echo "Pair-swap mode: ${PAIR_SWAP_MODE}"
echo "Pair source:    ${PAIR_SWAP_SOURCE_ENV}"
echo "Pair target:    ${PAIR_SWAP_TARGET_ENV}"
echo "RW holdout:     ${LEGO_HOLDOUT_MODE}"
echo "RW split mode:  ${LEGO_SPLIT_MODE}"
echo "RW root:        ${LEGO_DATASET_ROOT}"
echo "CSV suffix:     ${RESULT_CSV_SUFFIX}"
echo "Dry run:        ${DRY_RUN}"
echo "Skip existing:  ${SKIP_EXISTING}"
echo "Skip summary:   ${SKIP_SUMMARY}"
echo "=============================================="

"${CMD[@]}"
