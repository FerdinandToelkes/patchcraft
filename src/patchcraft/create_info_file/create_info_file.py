import sqlite3
import pandas as pd
import logging
import os

import patchcraft.utils as utils                            # import utility functions for progress bar
from patchcraft.create_info_file.config import Config       # import config class



def create_info_file_for_input_directory(input_config, output_config):
    """
    Create info file for slide directory.

    Args:
        input_config (dict): input config
        output_config (dict): output config
    """
    # start logging for info file
    utils.start_logging_for_info_file(input_config, output_config) 
    # setup dict to store metadata
    metadata = {attributes : [] for attributes in output_config['desired_metadata']}
    # get list of all files in the input directory
    filenames = os.listdir(input_config['path'])
    # filter out all files that are not sqlite files 
    filenames = [filename for filename in filenames if filename.endswith('.sqlite')]
    # iterate over all sqlite files and save metadata to metadata container
    for i, filename in enumerate(filenames):
        utils.printProgressBar(i, len(filenames), prefix = 'Progress:', suffix = 'Complete', length = 50)
        path = os.path.join(input_config['path'], filename)
        metadata['filename'].append(filename)
        get_metadata_from_sqlite_file(path, metadata)
    # switch metadata dict to pandas dataframe and save as csv file
    df = pd.DataFrame(metadata)
    path = os.path.join(output_config['path'], input_config['info_filename']) + '.csv'
    print("\nSaving info file to:", path)
    df.to_csv(path, index=False)


def get_metadata_from_sqlite_file(path: str, metadata: dict) -> dict:
    """
    Get metadata from sqlite file and save to metadata dict.

    Args:
        path (str): path to sqlite file
        metadata (dict): dict to store metadata

    Returns:
        dict: updated metadata dict
    """
    with sqlite3.connect(path) as conn:
        cursor = conn.cursor()
        for key in metadata.keys(): # list of the form ['diagnosis', 'stain', ...] self.output_config['saved_metadata']
            # skip filename since it is not stored in metadata table 
            if key == 'filename':
                continue
            # get value from metadata table
            cursor.execute("SELECT value FROM metadata WHERE key=?", (key,))
            row = cursor.fetchall()
            if row and row is not None:
                metadata[key].append(row[0][0])
            else:
                logging.debug(f"No {key} found in metadata table")
                metadata[key].append(None)
    
    

def main(args):
    # setup config and directory for all the output
    c = Config(args)
    c = c.setup_config()

    # create info file for slide directory
    create_info_file_for_input_directory(c['input'], c['output'])

    # print csv file
    path = os.path.join(c['output']['path'], c['input']['info_filename']) + '.csv'
    df = pd.read_csv(path)
    print()
    print("df_info:", df)

if __name__ == '__main__':
    main()
    