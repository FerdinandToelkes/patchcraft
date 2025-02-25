#!/bin/bash

# change the screen command depending on which data set you want to sample (see below for single examples of command without this script) 
# Usage: /sybig/home/ftoelkes/code/patchcraft/run_tile_sampling_for_whole_data_set.sh [num_jobs] [slides_per_job] [start_slide] [mode]
# Example: /sybig/home/ftoelkes/code/patchcraft/run_tile_sampling_for_whole_data_set.sh 4 33 0 train


# Input parameters with fallback to defaults
NUM_JOBS=${1:-4}                  # Default to 4 jobs
SLIDES_PER_JOB=${2:-40}           # Default to 20 slides per job
START_SLIDE=${3:-0}               # Default to start from slide 0
MODE=${4:-"train"}                # Default to 'train'

# Validate mode to avoid invalid directory errors
if [ "$MODE" != "train" ] && [ "$MODE" != "test" ]; then
  echo "Error: Unsupported mode '$MODE'. Supported modes are 'train' and 'test'."
  exit 1
fi

# Main loop
for ((i=0; i<NUM_JOBS; i++))
do
  SCREEN_NAME="sample_tiles_${MODE}_$i"
  DOCKER_NAME="ftoelkes_run_${MODE}_$i"
  JOB_START_SLIDE=$((START_SLIDE + i * SLIDES_PER_JOB))
  
  screen -dmS "$SCREEN_NAME" sh -c \
    "docker run --name $DOCKER_NAME -it -u \`id -u $USER\` --rm \
    -v /sybig/home/ftoelkes/code/patchcraft:/patchcraft \
    -v /sybig/projects/FedL/data:/data ftoelkes_patchcraft \
    ../generate_tiles.sh ../${MODE}_1024um_patches_marr \
    --input_path=/data/Marr_fixed_structure_only_ltds \
    --patch_size_um=1024 -pps all --mode '$MODE' \
    --white_threshold=0.25 -rp=1 --highest_zoom_level --overlap \
    --number_of_slides=$SLIDES_PER_JOB --start_slide=$JOB_START_SLIDE; exec bash"
done