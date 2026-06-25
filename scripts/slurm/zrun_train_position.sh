#!/bin/bash
#SBATCH -p rtx4090_batch
#SBATCH --nodelist c14
#SBATCH --gpus=1
#SBATCH --qos=batch
#SBATCH --cpus-per-task=10
#SBATCH --mem=64G
#SBATCH --ntasks=1
#SBATCH --job-name=position
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.log

#=============================================================================
# CONFIGURATION - Edit these variables to configure your training
#=============================================================================

# Training configuration
PERSPECTIVE="human"             # Options: camera, human
ENVIRONMENT="desert"           # Options: bridge, forest, desert, winter_town, city,
                                 #          bridge_2, forest_2, desert_2, winter_town_2, city_2
                                 #          bridge_3, forest_3, desert_3, winter_town_3, city_3
PROBE_TYPE="EfficientProbing"  # Options: EfficientProbing, ABMILPHead, ClassificationHead

# Activate conda environment
conda init bash
# Activate your environment (edit path to your conda install)
source "${CONDA_PREFIX_PATH:-$HOME/miniconda3}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV:-sparrta}"


#=============================================================================
# SETUP - Do not modify below this line
#=============================================================================

# Point the code at the data. Override these (or export them in your shell /
# sbatch --export) to match where you keep the SpaRRTa data and caches.
export SPARRTA_DATA_ROOT="${SPARRTA_DATA_ROOT:-/shared/results/common/kargin/unreal_engine/dataset/position_between_objects}"
export SPARRTA_LEGO_ROOT="${SPARRTA_LEGO_ROOT:-/shared/results/common/kargin/unreal_engine/dataset/position_between_objects/real_world_images}"
export SPARRTA_CACHE_DIR="${SPARRTA_CACHE_DIR:-./cache}"

# Export configuration as environment variables
export PERSPECTIVE
export ENVIRONMENT
export PROBE_TYPE
export HYDRA_FULL_ERROR=1

# Create logs directory if it doesn't exist
mkdir -p logs

# Print configuration
echo "=============================================="
echo "SLURM Job Configuration"
echo "=============================================="
echo "Job ID:       $SLURM_JOB_ID"
echo "Job Name:     $SLURM_JOB_NAME"
echo "Perspective:  $PERSPECTIVE"
echo "Environment:  $ENVIRONMENT"
echo "Probe Type:   $PROBE_TYPE"
echo "=============================================="

# Show GPU info
nvidia-smi -L

# Run the training
python scripts/run_sweep.py
