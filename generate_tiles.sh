#!/bin/bash

# this script is used for running a patchcraft sample_tiles on the server with all necessary preperation steps
# example of usage: 
# screen -dmS sample_tiles0 sh -c 'docker run --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code/patchcraft:/patchcraft -v /sybig/projects/FedL/data/sqlite:/data ftoelkes_patchcraft ../generate_tiles.sh  ../train_1024um_patches --input_path=/data --patch_size_um=1024 -pps all --mode "train" --white_threshold=0.25 -rp=1 --highest_zoom_level --overlap --number_of_slides=40 --start_slide=0; exec bash'

# get name for output directory
if [ -z "$1" ]; then
  output_dir="out"
  echo "No name for output directory given. Using default name 'out'."
else
  output_dir=$1
  echo "Using name '$output_dir' for output directory."
fi

# first create the out directory if it doesn't already exist
if [ ! -d "$output_dir" ]; then
  echo "Creating $output_dir directory..."
  mkdir $output_dir
fi

# run either the generate command in train or test mode depending on the first argument
echo "Run command: python3 -m patchcraft sample_tiles -o $output_dir ${@:2}"

python3 -m patchcraft sample_tiles -o $output_dir "${@:2}" # all arguments except the first one (the name of the output directory)

exit 0



