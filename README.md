# Patchcraft


## Outline

1. [Installation](#installation)
2. [Usage](#usage)
    - [Create an Information File](#create-an-information-file)
    - [Get an Overview of the Metadata](#get-an-overview-of-the-metadata)
    - [Sample train and test data as single files](#sample-train-and-test-data-as-single-files)
      - [Sample variable patch sizes](#sample-variable-patch-sizes)
      - [Sample tiles](#sample-tiles)
    - [Visualize files](#visualize-files)
3. [Setup with Docker](#setup-with-docker)
    - [Patchcraft with Docker](#patchcraft-with-docker)
4. [Bug Tracker](#bug-tracker)
5. [Configuration File](#configuration-file)


# Installation

The package was written using Python 3.11.3. Install Patchcraft with https or ssh by running one of the following commands:

```bash
pip3 install --upgrade git+https://github.com/FerdinandToelkes/patchcraft.git
pip3 install --upgrade git+ssh://git@github.com/FerdinandToelkes/patchcraft.git
```

or clone the repository locally and install with:

```bash
cd patchcraft
pip3 install -e .
```

# Usage

The newest version of this package does not contain the possibility of sampling blobs, i.e. .pt files which contain multiple patches from various slides. We found out that this restricts the usage of the package and in the end does not yield any speed improvements during data loading (it actually performed worse). Therefore, we decided to remove this option and only sample single files.

## Create an Information File

The *create_info_file* command is used for creating an **information file (default: metadata.csv) containing the metadata of all slides within the input directory.** The input directory should contain slides preprocessed by <a href="https://pamly.spang-lab.de/" target="_blank">Pamly</a> beforehand saved as .sqlite files. The code is built, assuming that the file names have the following format: 5 digit id, year, number of slide per patient (non negative integers), stain, diagnosis separated by "-" and ending with ".sqlite", e.g. '09120-1998-2-HE-DLBCL.sqlite' and Patchcraft takes the diagnosis and the stain out of the sqlite metadata table, if available. If not, one has to bring the data in the mentioned format to avoid any inconveniences. The output of the *create_info_file* command is a .csv file that can be used to get an overview of the metadata of the slides (see subsection [Get an Overview of the Metadata](#get-an-overview-of-the-metadata) below). The .csv file contains per default the following columns:

- **slide_name:** The name of the slide.
- **diagnosis:** The diagnosis of the slide.
- **stain:** The stain of the slide.

The command can be run with docker (see [Setup with Docker](#setup-with-docker)) as follows:

```bash
docker run -it --name ftoelkes_run1 -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code/patchcraft:/patchcraft -v /sybig/projects/FedL/data/sqlite:/data -v /sybig/home/ftoelkes/code/lymphoma:/lymphoma ftoelkes_patchcraft python3 -m patchcraft create_info_file --input_path=/data --output_path=/lymphoma
```

or locally with:

```bash
patchcraft create_info_file --input_path=PATH_TO_INPUT_DIR --output_path=PATH_TO_OUTPUT_FILE
```

For more information about the possible arguments, run the following command:

```bash
patchcraft create_info_file -h
```

The metadata.csv file can be used to get an overview of the metadata of the slides and to for example plot the distributions of the different classes. This is useful to get an overview of the data and to check if the data is balanced or not.

## Get an Overview of the Metadata

The *get_overview* command is used for getting a rough overview of the metadata of the slides within the input sqlite directory **by using the information file created beforehand.** The output is saved as a .yaml file (default: overview.yaml). It shows all unique values of the metadata columns (except the filename column) and the number of slides with these values. It takes the info .csv file as input that is generated with the command *create_info_file*. The command can be run as follows:

```bash
patchcraft get_overview --info_filename=PATH_TO_INFO_FILE
```

The contents will be printed to the console.
For more information about the possible arguments, use the '-h' flag.


## Sample train and test data as single files

The following two commands *sample_data* and *sample_tiles* are very similar. The first one focuses on sampling patches of any desired size from the Whole Slide Images (WSIs), while the second one samples the original tiles of the WSIs, that <a href="https://pamly.spang-lab.de/" target="_blank">Pamly</a> has divided the slide into.

### Sample varibale patch sizes

The *sample_data* command can be used to preprocess and, if necessary, augment patches from WSIs that have been preprocessed using <a href="https://pamly.spang-lab.de/" target="_blank">Pamly</a> and have the files follow the guidelines described in [Create an Information File](#create-an-information-file) (specifed structure of file name, and diagnosis + stain in sqlite metadata table). The patches are saved individually, i.e. each sampled patch is saved as a separate file containing the image itself with the diagnosis of the underlying WSI as a PyTorch tensor (provided that, the diagnosis is in the metadata table). The patches from a slide are stored in a directory named after the slide. A .log file containing the logging information of the command is also created and the config .yaml file used for sampling is also saved. The command can be executed as follows

```bash
patchcraft sample_data --input_path=/data --mode=train --number_of_slides=1 --number_of_patches_per_slide=10 --patch_size_um=200 --overlap=0.5 --number_of_repeated_patches=2 --perturbation_range=0.1 --no-highest_zoom_level
```

To sample all possible patches and/or to sample from all slides, the 'all' option can be used. The patches are sampled sequentially from coordinates that have been pre-computed so that the samples are equidistant and have the desired overlap. By requiring a non-zero perturbation range, these pre-computed coordinates are slightly perturbed, so that sampling a patch repeatedly adds some variety to the process. The flag *--no-highest_zoom_level* ensures faster sampling by making use of the quad tree structure of the WSIs, i.e., not using the smallest possible tiles to construct relatively big patches. This inturn leads to white areas being included in the patches.


There are many possible arguments for this command that can be given or written beforehand in a configuration .yaml file. The default configuration file is *config.yaml* (see [Configuration File](#configuration-file)). To get an overview of the possible arguments, use the '-h' flag.

Note: To sample patches close to the original tiles that a slide is divided into, set the overlap to zero and set 'patch_size_um' such that the resulting patches have the same size as the original tiles, e.g. for a tile size of 512 pixels and a resolution of 4000000 pixels per meter we compute:

patch_size_um = (512 px / 4000000 px/m) * 1e6 um/m = 128 micrometers

Alternatively one can just use the command introduced below. **After this sampling process, the single files may need further preperation to be used for the training of neural networks.** This is not done here. An example of possible usage of the single files is shown in <a href="https://gitlab.spang-lab.de/deep-learning/lymphoma" target="_blank">this project</a>.

### Sample tiles

The name 'tile' originates from the quad tree structure that has been used to save WSIs in a uniform format. The *sample_tiles* command can be used to sample tiles from WSIs that have been preprocessed as assumed in the preceding sections. The tiles are saved individually, i.e. each sampled tile is saved as a separate file containing the corresponding diagnosis as a PyTorch tensor. The tiles from a slide are stored in a directory named after the slide. A .log file containing the logging information of the command is also created and the config yaml file used for sampling is also saved. The command can be executed as follows

```bash
patchcraft sample_tiles --input_path=/data --mode=train --number_of_slides=1 --number_of_patches_per_slide=10 --patch_size_um=512 --overlap --perturbation_range=0.1 --highest_zoom_level
```

If necessary, the --help flag is again available to supply the user with more informations.

## View files

The *view_files* command can be used to plot 100 patches on a ten by ten grid, if desired with their diagnoses and stains. The command can be executed as follows

```bash
patchcraft view_files --path="path_to_files"
```

If necessary, the --help flag is again available to supply the user with more informations.



## Setup with Docker 

This package is meant to be run on servers since the data will probably not be stored locally. To use it on a server, start by cloning the repository *patchcraft* on the server and switch into it. To build the required Docker image, run the following command in the directory *patchcraft*:

```bash
docker build --tag ftoelkes_patchcraft -f Dockerfile .
```
where *ftoelkes_patchcraft* is an example name of the Docker image. To run the Docker image, run the following command:

```bash
docker run -it --name patchcraft_example -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code/patchcraft:/patchcraft -v /sybig/projects/FedL/data/sqlite:/data ftoelkes_patchcraft example_script.py
```

where *patchcraft_example* is an example name of the Docker container and */sybig/home/ftoelkes/code/patchcraft* is the path to the code directory on the host machine. We added some bash scripts to facilitate the sampling of big data sets, as described in the next section.


### Patchcraft with Docker

The commands needed for sampling tiles (arbitrary patches) are put together in a bash script called *generate_tiles.sh* (*generate.sh*). The script can be run with the following command:

```bash
screen -dmS sample_tiles0 sh -c 'docker run --name ftoelkes_run0 -it -u `id -u $USER` --rm -v /sybig/home/ftoelkes/code/patchcraft:/patchcraft -v /sybig/projects/FedL/data/sqlite:/data ftoelkes_patchcraft ../generate_tiles.sh  ../OUTPUT_PATH --input_path=/data --patch_size_um=1024 -pps all --mode "train" --white_threshold=0.25 -rp=1 --highest_zoom_level --overlap --number_of_slides=40 --start_slide=0; exec bash'
```

with *sample_tiles0* being the name of the screen session and the *OUTPUT_PATH* specifying the location of where all the output should be saved. The first argument is obligatory. 

To enter the screen session, run the following command:

```bash
screen -r sample_run
```

and to exit the screen session without stopping it, press *Ctrl + a* and then *Ctrl + d*.

In order to not having to copy and paste this command multiple times, the script *run_tile_sampling_for_whole_data_set.sh* can be used. It is a bash script that runs the *generate_tiles.sh* script multiple times 'in parallel' if you will. The script can be run with the following command:

```bash
/sybig/home/ftoelkes/code/patchcraft/run_tile_sampling_for_whole_data_set.sh NUM_JOBS SLIDES_PER_JOB START_SLIDE MODE
```

where *NUM_JOBS* is the number of parallel jobs, *SLIDES_PER_JOB* is the number of slides per job, *START_SLIDE* is the number of the first slide to be processed and *MODE* is either 'train' or 'test'. The script will create a screen session for each job and run the *generate_tiles.sh* script with the specified arguments. 

The advantage of running the code more like a script than a package is that the Docker image has to be build only once. The file *requirements.txt* is exclusively needed when using *patchcraft* as a script. All files generated in any of the mounted directories, will be directly visible on the host machine.




## Bug Tracker

If you run into any kind of unusual behavior while using the package please open an issue and report as detailed as possible what the problem is and under which condtions it occurs.


## Configuration File

The configuration file is a .yaml file containing the parameters for the different commands. The default configuration file is called *config.yaml*. The following is an example of a configuration file:

```yaml 

input:
  # Path where the data is located.
  path: '/data' 
  # prefix for all the names of the files resulting from command create_info_file (csv and log file)
  info_filename: 'metadata' 
  # Path to the file containing the overview of the metadata of all slides -> is the only file with prefix 'overview' 
  overview_filename: 'overview.yaml'
  

output:
  # Output path where processed data will be stored. 
  path: './out'                                 
  # Mode of the sampling process. (train or test)
  mode: 'train'                                 
  # Number of slides to be processed -> useful for testing sampling process.
  number_of_slides: 1
  # Number of patches per slide -> useful for testing sampling process.
  number_of_patches_per_slide: 100              # only for trainings mode 
  # Number of slide to start with -> for "parallelization" of the sampling process.
  start_slide: 0                              
  # Number of times the same patch should be sampled from the same slide
  number_of_repeated_patches: 1                 # only for trainings mode   
  # The threshold for the white ratio of the patch. If the white ratio is higher than the threshold, the patch will be discarded.
  white_threshold: 0.25                          # only for trainings mode     
  # Metadata of the slide (directly from sqlite file) that will be saved in the csv file. (The coordinates of the patch will be added automatically)
  desired_metadata: ['filename', 'diagnosis', 'stain'] 
  # log level
  log_level: 'INFO'

  

  
# Parameters needed for generating both training or test data
general_transforms:
  # Parameters for sampling
  sampling:
    # The stain to be sampled 
    stain: 'HE'
    # The patch size in micrometers to be sampled.
    patch_size_um: 128
    # Weather to use overlap=0.5 in sample_tiles
    overlap_bool: False # only for sample_tiles command
    # set how much two neighboring patches should overlap (in percentage)
    overlap: 0.5 # only for sample_data command
    # Pixel resolution per m of the WSI (default value if no resolution is available)
    wsi_pixels_per_m: 4000000  
    # random seed for making sampling more reproducible, default value is 0 -> use date + time as seed
    random_seed: 0
    # Target size for resizing. (Input tensor size for the net, tensors should be [N, target_size, target_size])
    target_size: 224 # default from image net, only needed for sample_data command
    # Whether to sample from highest level or not. (Sampling from highest level is disabled)
    highest_zoom_level: False


# Parameters needed only for generate_train_data (mostly for augmentation) 
training_transforms:
  sampling:
    # patch size ranges, sampled sizes will be in (1+size_range, 1-size_range)* patch_size          
    patch_size_range: 0.1 # only for sample_data command
    # perturbation range for the patch coordinates in percentage of the tile size (note that the maximium perturbation is 0.5)
    perturbation_range: 0.1 # only for sample_data command
  
  rotation: # only for sample_data command
    # Enable rotation in the pipeline. (Rotation is enabled)
    enabled: True  

  flips:
    # Enable flipping in the pipeline. (Flipping is enabled)
    enabled: True  

  color_jitter:
    # random brightness changes in range [0,brightness_jitter]
    brightness_jitter: 0.10
    # random contrast changes in range [0,contrast_jitter]   
    contrast_jitter: 0.10
    # random saturation changes in range [0,saturation_jitter]
    saturation_jitter: 0.10
    # random hue changes in range [-hue_jitter,hue_jitter]
    hue_jitter: 0.025
  
```

