import pandas as pd
import yaml                           # for saving the overview as a yaml file
import math                           # for checking if a value is nan
import os                             # for checking if the info file exists

from .config import Config            # for the config class

# can be later updated to tell how many different datasets, how many needle biopsies etc.

def get_overview(input_config: dict, output_config: dict, path_to_info_file: str) -> None:
    """ Get an overview of the contents of the info file. 
    
    Args:
        input_config (dict): input config
        output_config (dict): output config
        path_to_info_file (str): path to the info file
    """
    # setup
    df = pd.read_csv(path_to_info_file)
    metadata = output_config['desired_metadata']
    # get all unique values for the metadata
    unique_vals = get_unique_values(df, metadata)
    # get how often each unique value occurs in the metadata
    overview = get_occurances(df, unique_vals)
    # check if occurences add up to the number of rows in the dataframe
    for md in overview.keys():
        if df.shape[0] != sum(overview[md].values()):
            raise Exception("The occurences of the metadata do not add up to the number of rows in the dataframe.") 
    # nicely print the contents of overview
    print_overview(df, overview)
    # save the overview as a yaml file
    path = os.path.join(output_config['path'], input_config['overview_filename'])
    with open(path, 'w') as file:
        yaml.dump(overview, file)
        print(f"Saved the overview as a yaml file in {path}")


######################################################################################################################################
########################################### Utility function for get_overview() ######################################################
###################################################################################################################################### 

def get_unique_values(df: pd.DataFrame, metadata: list) -> dict:
    """ Get all unique values for the metadata (e.g. which different stains exist etc) and save it in a dict.
    
    Args:
        df (pd.DataFrame): The dataframe containing the metadata.
        metadata (list): List of metadata attributes.

    Returns:
        dict: The unique values for each metadata attribute.
    """
    unique_vals = {}
    for md in metadata:
        if md == 'filename': # has no important information
            continue
        # get all unique values for the metadata as a list instead of an array
        unique_vals[md] = df[md].unique().tolist() 
    return unique_vals

def get_occurances(df: pd.DataFrame, unique_vals: dict) -> dict:
    """ Get how often each unique value occurs in the metadata and save it in a dict. 
    
    Args:
        df (pd.DataFrame): The dataframe containing the metadata.
        unique_vals (dict): The unique values for each metadata attribute.

    Returns:
        dict: How often each unique value occurs in the
    """
    overview = {}
    for md in unique_vals.keys():
        overview[md] = {}
        for val in unique_vals[md]:
            # check if the value is a string or a number
            if isinstance(val, str):
                overview[md][val] = df[df[md] == val].shape[0]
            elif math.isnan(val):
                overview[md][val] = df[df[md].isnull()].shape[0]
            else:
                raise Exception(f"Unexpected type of metadata value: {type(val)}")
    return overview

def print_overview(df: pd.DataFrame, overview: dict):
    """ Print the contents of overview and the head of the dataframe.
    
    Args:
        df (pd.DataFrame): The dataframe containing the metadata.
        overview (dict): How often each unique value occurs in the metadata.
    """
    print("Head of the info file:")
    print(df.head())
    print()
    print("Overview of the info file:")
    nan_count = 0
    for md in overview.keys():
        print()
        print(md)
        for val in overview[md].keys():
            # check if the value is a string or a number -> print nan at the end
            if isinstance(val, str):
                print(f"    {val}: {overview[md][val]}")
            else:
                nan_count = overview[md][val]
        print(f"    nan: {nan_count}")  
    
    
######################################################################################################################################

def main(args):
    """ Main function of the get_overview subcommand. 
    
    Args:
        args (argparse.Namespace): Arguments given in the command line.
    """
    # set up the config and a directory for all the output
    c = Config(args)
    c = c.setup_config()

    # check if the info file exists in the specified path
    path_to_info_file = os.path.join(c['output']['path'], c['input']['info_filename']) + '.csv'
    if not os.path.exists(path_to_info_file):
        print("path:", path_to_info_file)
        raise Exception(f"Info file does not exist in the specified path: {path_to_info_file}. Use the 'create_info_file' command to create it.")

    # get an overview of the contents of the info file
    get_overview(c['input'], c['output'], path_to_info_file)



if __name__ == '__main__':
    main()
    