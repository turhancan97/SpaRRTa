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
#SBATCH --job-name=lego_rebuttal
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

#=============================================================================
# CONFIGURATION - Edit these variables if needed
#=============================================================================
OUTPUT_DIR="result_lego_rebuttal"
EPOCHS=150
WARMUP_EPOCHS=15
BATCH_SIZE=16
BACKBONES="clip_b16_laion" #dino_b16,dinov2_b14,dinov2_b14_reg,dinov2_l14_reg,dinov3_b16,dinov3_timm,croco_b16,crocov2_b16,mae_b16,maskfeat_vitb16,spa_b16,spa_l16,vggt_l16,deit3_b16,clip_b16_laion"
HEADS="EfficientProbing" #EfficientProbing,ABMILP,GAP
SEEDS="8,42,123" #8,42,123
SPLIT_MODES="time_series" # random, time_series, or add clip_block_random,clip_block_time_series when needed
CLIP_BLOCK_SIZE=20
CLIP_ID_SOURCE="filename_numeric" # filename_numeric, parent_folder
SKIP_EXISTING=false      # true/false
SKIP_SUMMARY=false      # true/false
RERUN_SEED8_FINAL=true # true/false
DRY_RUN=false           # true/false

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
conda activate "${CONDA_ENV:-sparrta}"

mkdir -p logs

CMD=(
  python scripts/run_lego_rebuttal.py
  --output-dir "${OUTPUT_DIR}"
  --epochs "${EPOCHS}"
  --warmup-epochs "${WARMUP_EPOCHS}"
  --batch-size "${BATCH_SIZE}"
  --backbones "${BACKBONES}"
  --heads "${HEADS}"
  --split-modes "${SPLIT_MODES}"
  --seeds "${SEEDS}"
  --clip-block-size "${CLIP_BLOCK_SIZE}"
  --clip-id-source "${CLIP_ID_SOURCE}"
)

if [ "${SKIP_EXISTING}" = "true" ]; then
  CMD+=(--skip-existing)
fi
if [ "${SKIP_SUMMARY}" = "true" ]; then
  CMD+=(--skip-summary)
fi
if [ "${RERUN_SEED8_FINAL}" = "true" ]; then
  CMD+=(--rerun-seed8-final)
fi
if [ "${DRY_RUN}" = "true" ]; then
  CMD+=(--dry-run)
fi

echo "=============================================="
echo "SLURM Job Configuration"
echo "=============================================="
echo "Job ID:            ${SLURM_JOB_ID}"
echo "Job Name:          ${SLURM_JOB_NAME}"
echo "Output dir:        ${OUTPUT_DIR}"
echo "Epochs:            ${EPOCHS}"
echo "Warmup epochs:     ${WARMUP_EPOCHS}"
echo "Batch size:        ${BATCH_SIZE}"
echo "Backbones:         ${BACKBONES}"
echo "Heads:             ${HEADS}"
echo "Split modes:       ${SPLIT_MODES}"
echo "Clip block size:   ${CLIP_BLOCK_SIZE}"
echo "Clip id source:    ${CLIP_ID_SOURCE}"
echo "Skip existing:     ${SKIP_EXISTING}"
echo "Skip summary:      ${SKIP_SUMMARY}"
echo "Rerun seed8 final: ${RERUN_SEED8_FINAL}"
echo "Dry run:           ${DRY_RUN}"
echo "=============================================="

"${CMD[@]}"
