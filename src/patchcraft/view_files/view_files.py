import matplotlib.pyplot as plt         # for plotting
import torch                            # for loading the tensors
import pandas as pd                     # for loading the metadata
import argparse                         # for command line arguments (only for type hints)
import os                               # for checking if files exist

from .config import Config              # for the config class
from patchcraft.diagnosis_maps import LABELS_MAP_INT_TO_STRING 


def plot_100_patches_from_one_slide(args: argparse.Namespace, output_config: dict, labels_map_int_to_string: dict):
    """ Plot 100 patches from a slide. 
    
    Args:
        args (argparse.Namespace): The command line arguments.
        output_config (dict): The output config from the config file.
        labels_map_int_to_string (dict): A dictionary mapping the labels to their names.
    """
    # check if pt files exist in the given directory
    filenames = os.listdir(args.path)
    filenames = [os.path.join(args.path,filename) for filename in filenames if filename.endswith('.pt')]
    if len(filenames) == 0:
        raise Exception(f"No .pt files could be found in the specified directory {args.path}. You could switch to the directory and run the command without specifying the path.")
    # sort the filenames using the patch number (filename format: patch_123_coord_((37,44),(12,13)).pt)
    try:
        filenames = sorted(filenames, key=lambda x: int(x.split('_')[1]))
    except:
        raise Exception("The filenames could not be sorted. Try to call the command within the output directory, e.g. --path=./84433-2012-0-HE-DLBCL")
    if args.index + 100 > len(filenames):
        print("The index you specified is too high. This could lead to an index out of bounds error or other unwanted behaviour.")
    # Load 100 tensors + labels from the .pt files
    tensors = []
    labels = []
    for filename in filenames[args.index : args.index+100]:
        try:
            data = torch.load(filename)
            tensors.append(data[0])
            labels.append(data[1])
        except:
            raise Exception(f"The .pt file {filename} could not be loaded.")

    # Load the metadata and select the metadata for the 100 patches
    metadata = pd.read_csv(os.path.join(args.path, "metadata.csv"))
    metadata = metadata.iloc[args.index : args.index+100]
    # Plot the 100 patches with or without metadata
    if metadata is not None:
        print("Metadata found. Plotting the patches with metadata.")
        fig = plot_with_metadata(tensors, metadata)
    else:
        print("No metadata could be found. Plotting the patches without metadata.")
        fig = plot_without_metadata(tensors, labels, labels_map_int_to_string)
    # Save the plot as a PNG file with the specified DPI
    path = os.path.join(output_config['path'], args.output_name)
    plt.savefig(path, dpi=args.dpi)
    print(f"Saved the plot to {path}.")
    # Close the figure  
    plt.close(fig)

################################################## Utility functions ##################################################

def plot_with_metadata(tensors: list, metadata: pd.DataFrame) -> plt.Figure:
    """ Plot the tensors with metadata.

    Args:
        tensors (list): A list of tensors.
        metadata (pd.DataFrame): A DataFrame containing the metadata for the tensors.

    Returns:
        fig: The figure object.
    """
    # Set up the figure and axes
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i, (tensor, md) in enumerate(zip(tensors, metadata.values)):
        ax = axes[i % 10, i // 10]
        tensor_np = tensor.numpy()
        ax.imshow(tensor_np.transpose(1, 2, 0))
        ax.axis("off")
        _, _, _, coords = md # filename, diagnosis, stain, coords
        ax.text(0, 0, f"Coords: {coords}", fontsize=8)
    return fig

def plot_without_metadata(tensors: list, labels: list, labels_map_int_to_string: dict) -> plt.Figure:
    """ Plot the tensors without metadata.

    Args:
        tensors (list): A list of tensors.
        labels (list): A list of labels.
        labels_map_int_to_string (dict): A dictionary mapping the labels to their names.

    Returns:
        fig: The figure object.
    """
    # Set up the figure and axes
    fig, axes = plt.subplots(10, 10, figsize=(20, 20))
    for i, (tensor, label) in enumerate(zip(tensors, labels)):
        ax = axes[i % 10, i // 10]
        tensor_np = tensor.numpy()
        ax.imshow(tensor_np.transpose(1, 2, 0))
        ax.axis("off")
        ax.text(0, 0, f"Label: {labels_map_int_to_string[label.item()]}", fontsize=8)
    return fig

###########################################################################################################################################

def main(args: argparse.Namespace):
    """ Main function of the view subcommand. 
    
    Args:
        args (argparse.Namespace): The command line arguments.
    """
    # set up the config and a directory for all the output
    c = Config(args)
    c = c.setup_config()

    # Plot 100 patches from the blob -> args and config as input since not all parameters are in the config file
    plot_100_patches_from_one_slide(args, c['output'], LABELS_MAP_INT_TO_STRING)

if __name__ == "__main__":
    main()