#!/bin/bash

# this script is used for running a patchcraft sample_data on the server with all necessary preperation steps
# sample train data
# screen -dmS ftoelkes sh -c 'docker run -it -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code/patchcraft:/patchcraft -v /sybig/projects/FedL/data/sqlite:/data ftoelkes_patchcraft ../generate.sh train_200um_patches --input_path=/data --mode=train --number_of_slides=10 --start_slide=0 --number_of_patches_per_slide=all --patch_size_um=200 --overlap=0.5 --number_of_repeated_patches=4 --perturbation_range=0.1 --no-highest_zoom_level; exec bash'

# sample test data
# screen -dmS sample_patches_100um sh -c 'docker run --name ftoelkes_run1 -it -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code/patchcraft:/patchcraft -v /sybig/projects/FedL/data/sqlite:/data fto_patchcraft ../generate.sh test_100um_patches --input_path=/data --mode=test --number_of_slides=1 --start_slide=0 --number_of_patches_per_slide=all --patch_size_um=100 --overlap=0.0 --number_of_repeated_patches=1 --perturbation_range=0.0 --no-highest_zoom_level; exec bash'

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
echo "Run command: python3 -m patchcraft sample_data -o $output_dir ${@:2}"
python3 -m patchcraft sample_data -o $output_dir "${@:2}" # all arguments except the first one (the name of the output directory)

exit 0
