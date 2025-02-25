import os                                             # for file and directory operations
import logging                                        # for logging errors, warnings, and info messages
import torch          
import numpy as np  
import pandas as pd
import time
import sqlite3
import yaml
from PIL import Image
from io import BytesIO
from numpy import random 

from patchcraft.diagnosis_maps import LABELS_MAP_STRING_TO_INT

import patchcraft.sample_tiles.augment as augment       # needed for to_torch_tensor and resize functions
import patchcraft.utils as utils                                # import utility functions for logging and progress bar



class GenerateData(): 
    """ Class for preprocessing patches from whole slide images. """
    def __init__(self, config: dict):
        """
        Initialize the Preprocessing class. 

        Args:
            config (dict): Dictionary containing the configuration parameters.
        """
        # setup often used config parameters
        self.config = config
        self.input_config = config['input']
        self.output_config = config['output']
        self.general_transform_config = config['general_transforms'] 
        self.training_transform_config = config['training_transforms']
        self.mode = self.output_config['mode'] 
        
        # setup labels for the different classes -> will be used for switching from string to int labels
        self.labels_map_string_to_int = LABELS_MAP_STRING_TO_INT

    def preprocess_one_patch(self, path: str, current_coords: list, level_to_sample_from: int) -> tuple:
        """
        Preprocess one patch from a slide.

        Args:
            path (str): The path to the slide.
            current_coords (list): A list containing the coordinates at the level_to_sample_from (from sqlite database).
            level_to_sample_from (int): The zoom level from which the tiles should be sampled.

        Returns:
            tuple: A tuple containing the preprocessed patch and the coordinates of the patch.
        """
        # check if we have to sample one or multiple tiles to obtain the patch
        if len(current_coords) == 2: # just a tuple with x and y coordinate of tile
            one_patch_np, tile_coord = self.get_tile_and_prepare_as_patch(path, current_coords, level_to_sample_from)
        else:
            one_patch_np, tile_coord = self.get_tiles_and_combine_to_patch(path, current_coords, level_to_sample_from)
        
        # if there was at least one tile missing, None was returned
        if one_patch_np is None: 
            return None, None # to be consistent with return type of preprocess_one_patch 
        logging.debug(f'current_coords: {current_coords}')
        logging.debug(f'one_patch_np.shape: {one_patch_np.shape}')
        return one_patch_np, tile_coord
    
    def preprocess_patches_from_one_slide(self, metadata : dict, slide_name: str, output_dir_for_patches_of_slide: str):
        """
        Sample patches from one slide and update the metadata dict.

        Args:
            metadata (dict): Dict containing the metadata that should be stored.
            slide_name (str): The name of the slide.
            output_dir_for_patches_of_slide (str): The path to the output directory for patches of this slide.
        """
        # setup path to the slide
        path_to_slide = os.path.join(self.input_config['path'], slide_name)

        # get parameters of the slide and check if we can preprocess patches from this slide
        coords_for_sampling, level_to_sample_from, metadata_of_one_slide = self.get_parameters_of_slide(path_to_slide) 
        # if there was a problem with the parameters, None was returned
        if coords_for_sampling is None or level_to_sample_from is None or self.patch_size_in_pixels is None or metadata_of_one_slide is None:
            logging.warning(f'Some parameters for slide {path_to_slide} could not be found, returning None')
            return None, None # to be consistent with the return type within this function
        
        # switch from string to int labels, get the length of coords_for_sampling and the label for the slide
        self.switch_string_to_int_label(metadata_of_one_slide) # i.e. metadata_of_one_slide['diagnosis'] -> int
        label_for_slide = metadata_of_one_slide['diagnosis']
        label_for_slide = torch.tensor(label_for_slide)

        # iterate through all coordinates = (tile coordinates, patch coordinates) and preprocess one patch at a time
        len_coords_for_sampling = len(coords_for_sampling) if self.output_config['number_of_patches_per_slide'] == 'all' else min(len(coords_for_sampling), int(self.output_config['number_of_patches_per_slide']))
        sampling_possible = False
        for coords in coords_for_sampling: # coords are only tile_coords
            # break if we have sampled the desired number of patches
            if self.output_config['number_of_patches_per_slide'] != 'all' and self.processed_patches >= int(self.output_config['number_of_patches_per_slide']):
                logging.info(f'Number of processed patches ({self.processed_patches}) reached the number of patches to be sampled ({self.output_config["number_of_patches_per_slide"]}) for slide {path_to_slide}, continuing with next slide')
                print(f"Number of processed patches ({self.processed_patches}) reached the number of patches to be sampled ({self.output_config['number_of_patches_per_slide']}) for slide {path_to_slide}, continuing with next slide")
                break
            for _ in range(self.output_config['number_of_repeated_patches']):
                # preprocess one patch if possible 
                one_patch_np, tile_coord = self.preprocess_one_patch(path_to_slide, coords, level_to_sample_from)
                if isinstance(one_patch_np, np.ndarray):
                    # augment patch if in training mode and save it to file
                    self.finish_preprocessing_patch(output_dir_for_patches_of_slide, metadata_of_one_slide, tile_coord, metadata, one_patch_np, label_for_slide)
                    logging.debug(f'Preprocessed patch {self.processed_patches} from slide {path_to_slide}')
                    print(f"(Processed patches: {self.processed_patches} & (Processed patches / repeated patches) / possible coordinates of the slide: {int(self.processed_patches/self.output_config['number_of_repeated_patches'])}/{len_coords_for_sampling}", end='\r')
                    self.processed_patches += 1
                    # from this coordinate we can sample multiple patches
                    sampling_possible = True
                elif one_patch_np is None:
                    sampling_possible = False
                    logging.debug(f'Not enough tiles found in tiles table for slide {path_to_slide} for a certain patch')
                    continue
                else:
                    sampling_possible = False
                    logging.error(f'Unknown error occured while preprocessing patches from slide {path_to_slide}')

                # continue with next coordinate if sampling was not possible
                if sampling_possible is False:
                    break


    def preprocess_patches(self):
        """ Preprocess patches from multiple slides and save them as single files. """
        # get all slide names from the input directory
        slide_names = os.listdir(self.input_config['path'])
        # select n slides with correct stain starting at start_slide (n=number of slides)
        stain = self.general_transform_config['sampling']['stain']
        slide_names = [filename for filename in slide_names if filename.endswith('.sqlite') and stain in filename]
        slide_names.sort() # for reproducibility
        slide_names = slide_names[self.output_config['start_slide']-0:] # count from 0
        if self.output_config['number_of_slides'] != 'all':
            slide_names = slide_names[:int(self.output_config['number_of_slides'])]
        # check if all slides share the same tile size and resolutions and return them if they do
        self.tile_size, resolutions = self.check_and_get_tile_size_and_resolutions(slide_names)
        self.patch_size_in_pixels = self.get_patch_size_in_pixels(resolutions) 
        logging.debug(f'patch_size_in_pixels: {self.patch_size_in_pixels}')

        # iterate through the given number of slides and preprocess patches from each slide
        for i, slide_name in enumerate(slide_names):
            # reset the metadata dict for each slide; the filename is the one of the patch
            metadata = {'filename': [], 'diagnosis': [], 'stain': [], 'coordinates': []} # 'dataset:survival': [],
            print()
            print(f"Processing slide {slide_name} which is slide number {i+1}/{len(slide_names)} of this run.", end='\r')
            print()
            logging.debug(f"Processing slide {slide_name} which is slide number {i+1} of this run.")
            
            # create output directory for patches of this slide
            name = slide_name.split('.')[0] # remove .sqlite from slide_
            output_dir_for_patches_of_slide = os.path.join(self.output_config['path'], name)
            if os.path.exists(output_dir_for_patches_of_slide):
                logging.info(f"Output directory {output_dir_for_patches_of_slide} already exists, continuing with next slide")
                continue
            else:
                os.makedirs(output_dir_for_patches_of_slide, exist_ok=False)

            # reset the number of processed patches and start logging for this slide
            self.processed_patches = 0
            utils.start_logging_for_slide(self.output_config, self.general_transform_config, self.input_config['path'], output_dir_for_patches_of_slide) 

            # preprocess patches from one slide and save them as single files to the directory 'output_dir_for_patches_of_slide'
            self.preprocess_patches_from_one_slide(metadata, slide_name, output_dir_for_patches_of_slide) # this also updates the metadata dict
            print(f"Preprocessed {self.processed_patches} patches from slide {slide_name} continuing with next slide", end='\r')
    
            # save metadata seperately (here we also check if number of pt files is the same as the length of csv file)
            self.save_metadata(metadata, output_dir_for_patches_of_slide)
            self.save_config()

        print()
        print("Finished preprocessing patches from all slides")

    ###################################################################################################################################
    ################################################### Utility functions #############################################################
    ###################################################################################################################################  

    def check_and_get_tile_size_and_resolutions(self, slide_names: list) -> tuple:
        """ Check if all slides share the same tile size and resolutions and return them if they do. 
        
        Args:
            slide_names (list): A list containing the names of the slides.

        Returns:
            tuple: A tuple containing the tile size and the resolutions.
        """
        tile_sizes = []
        resolutions = []
        for slide_name in slide_names:
            path_to_slide = os.path.join(self.input_config['path'], slide_name)
            tile_size = self.get_tile_size_from_database(path_to_slide)
            resolutions_of_slide = self.get_resolutions_from_database(path_to_slide)
            tile_sizes.append(tile_size)
            resolutions.append(resolutions_of_slide)
        if len(set(tile_sizes)) > 1:
            logging.error(f'Tile sizes of the slides {slide_names} are not the same: {tile_sizes}')
            raise Exception(f'Tile sizes of the slides {slide_names} are not the same: {tile_sizes}')
        if len(set(resolutions)) > 1:
            logging.error(f'Resolutions of the slides {slide_names} are not the same: {resolutions}')
            raise Exception(f'Resolutions of the slides {slide_names} are not the same: {resolutions}')
        logging.info(f'Tile sizes of the slides {slide_names} are the same: {tile_sizes[0]}')
        logging.info(f'Resolutions of the slides {slide_names} are the same: {resolutions[0]}')
        if tile_sizes[0] is not None and resolutions[0] is not None:
            return tile_sizes[0], resolutions[0]
        else:
            logging.error(f'Tile size or resolutions could not be found for slides {slide_names}')
            raise Exception(f'Tile size or resolutions could not be found for slides {slide_names}')

    
    def save_config(self):
        """ Save the config of the current run to the output directory as a yaml file. """
        # save config of current run to output directory for patches of this slide
        filename_yaml = os.path.join(self.output_config['path'], 'config.yaml')
        with open(filename_yaml, 'w') as file:
            yaml.dump(self.config, file)
        logging.info(f'Config of this run saved to "{filename_yaml}"')

    
    def get_number_of_processed_patches(self, output_dir_for_patches_of_slide: str) -> int:
        """
        Get the highest number assigned to a patch from the output directory for patches of a slide. Filenames are of the form 'patch_{number}_coords_{coordinates}.pt'.

        Args:
            output_dir_for_patches_of_slide (str): The path to the output directory for patches of this slide.

        Returns:
            int: The number of processed patches.
        """
        # get all files in the output directory
        files = os.listdir(output_dir_for_patches_of_slide)
        # filter out non pt files 
        files = [filename for filename in files if filename.endswith('.pt')]
        # get the number following 'patch_' in the filename
        numbers = [int(filename.split('_')[1]) for filename in files] # .split('_') -> ['patch', '{number}', 'coords', '{coordinates}.pt']
        # get the highest number
        number_of_processed_patches = max(numbers)
        return number_of_processed_patches
    
    def get_tile_and_prepare_as_patch(self, path: str, current_coords: tuple[int, int], level_to_sample_from: int) -> tuple:
        """
        Retrieves tile from the database and brings it into the right shape.

        Args:
            path (str): Path to the SQLite database.
            current_coords (tuple): Coordinates of the tile which should be prepared as a patch.
            level_to_sample_from (int): Zoom level from which the tiles should be sampled.

        Returns:
            tuple: A tuple containing the finished patch and the coordinates of the patch.
        """
        try:
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()
        except:
            logging.error("Error occurred while connecting to sqlite database")
            raise Exception("Error occurred while connecting to sqlite database")
        else:
            cursor.execute("SELECT jpeg FROM tiles WHERE x=? AND y=? AND level=? LIMIT 1", (current_coords[0], current_coords[1], level_to_sample_from))
            row = cursor.fetchone()
            if row is not None:
                # Extract the image data from the row
                jpeg_data = row[0]
                # Create a BytesIO object from the image data
                with BytesIO(jpeg_data) as bytes_io:
                    # Read the image from the BytesIO object using PIL.Image.open()
                    with Image.open(bytes_io) as pil_image: # returns an image array in NumPy format
                        # Convert the PIL image to a NumPy array and append to y_tiles
                        patch = np.array(pil_image)
                cursor.close()
            else:
                return None, None
  
        # Note: numpy dimensions (matrix) neq x,y dimensions (coordinates) -> switch axis 0 and 1 before slicing
        patch = np.transpose(patch, (1,0,2))
        logging.debug(f'tiles_for_one_patch.shape before: {patch.shape}')
        # get the corresponding tile coord of underlying (bigger) tile 
        tile_coord = (current_coords[0], current_coords[1], 1)  
        return patch, tile_coord

    def get_tiles_and_combine_to_patch(self, path: str, current_coords: tuple[int, int], level_to_sample_from: int) -> tuple:
        """
        Retrieve tiles from the database and combine them into a patch.

        Args:
            path (str): Path to the SQLite database.
            tile_coords (tuple): Coordinates of the tiles which should be combined to a patch.
            level_to_sample_from (int): Zoom level from which the tiles should be sampled.

        Returns:
            tuple: A tuple containing the finished patch and the coordinates of the patch.
        """
        # get the starting point and the range of the coordinates
        starting_point = current_coords[0]
        xy_range = np.sqrt(len(current_coords)) if type(current_coords) == list else 1
        if xy_range % 1 != 0:
            logging.error(f"y_range is not a perfect square: {xy_range} , raise exception")
            raise Exception(f"y_range is not a perfect square: {xy_range}")
        xy_range = int(xy_range)
        
        logging.debug(f'starting_point: {starting_point}, y_range: {xy_range}')

        tiles_for_one_patch = []
        try:
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()
        except:
            logging.error("Error occurred while connecting to sqlite database")
            raise Exception("Error occurred while connecting to sqlite database")
        else:
            for xi in range(starting_point[0], starting_point[0] + xy_range):
                # y_tiles = self.get_y_tiles_for_one_x(cursor, xi, starting_point[1], y_plus, y_minus, level_to_sample_from)
                y_tiles = self.get_y_tiles_for_one_x(cursor, xi, starting_point[1], xy_range, level_to_sample_from)
                
                # Concatenate the tiles in the y-direction
                y_tiles = np.concatenate(y_tiles, axis=0) 
                tiles_for_one_patch.append(y_tiles)
            cursor.close()

        # Concatenate the tiles in the x direction
        tiles_for_one_patch = np.concatenate(tiles_for_one_patch, axis=1) 
        # Note: numpy dimensions (matrix) neq x,y dimensions (coordinates) -> switch axis 0 and 1 before slicing
        tiles_for_one_patch = np.transpose(tiles_for_one_patch, (1,0,2))
        logging.debug(f'tiles_for_one_patch.shape before: {tiles_for_one_patch.shape}')
        # check if the patch is relevant, especially important for overlapping patches
        if self.check_if_tile_is_relevant(tiles_for_one_patch, self.output_config['white_threshold']) is False:
            return None, None
        logging.debug(f'tiles_for_one_patch.shape after:  {tiles_for_one_patch.shape}') 
        # get the corresponding tile coord of underlying (bigger) tile 
        tile_coord = (starting_point[0], starting_point[1], xy_range)  
        return tiles_for_one_patch, tile_coord
    

    #################################### Utility functions for get_tiles_and_combine_to_patch() ##########################################
    
    def get_y_tiles_for_one_x(self, cursor: sqlite3.Cursor, xi: int, starting_y: int, xy_range: int, level_to_sample_from: int) -> list:
        """
        Retrieve tiles from the database for one x-coordinate and combine them into a list.

        Args:
            cursor (sqlite3.Cursor): Cursor to execute SQL queries.
            xi (int): current x-coordinate.
            starting_y (int): starting y-coordinate.
            xy_range (int): number of tiles in positive x-direction (starting tile is excluded).
            level_to_sample_from (int): Zoom level from which the tiles should be sampled.

        Returns:
            list: A list containing the tiles in the y-direction.
        """
        y_tiles = []
        # Iterate through the necessary tiles in the y-direction
        for yi in range(starting_y, starting_y + xy_range):
            # get the jpeg data from the database
            cursor.execute("SELECT jpeg FROM tiles WHERE x=? AND y=? AND level=? LIMIT 1", (xi, yi, level_to_sample_from))
            row = cursor.fetchone()
            if row is not None:
                # Extract the image data from the row
                jpeg_data = row[0]
                # Create a BytesIO object from the image data
                with BytesIO(jpeg_data) as bytes_io:
                    # Read the image from the BytesIO object using PIL.Image.open()
                    with Image.open(bytes_io) as pil_image: # returns an image array in NumPy format
                        # Convert the PIL image to a NumPy array and append to y_tiles
                        img = np.array(pil_image)
                        y_tiles.append(img)
            else:
                # append white image if no tile is found
                y_tiles.append(np.ones((self.tile_size, self.tile_size, 3), dtype=np.uint8)*255) 
        return y_tiles
    
    
    ###################################################################################################################################
    
    def save_metadata(self, metadata: list, output_dir_for_patches_of_slide: str):
        """
        Save the metadata as csv file for the patches preprocessed from current slide.

        Args:
            metadata (list): A list containing the metadata of the form [[('filename', filename), ('diagnosis', diagnosis), ...], [...], ...].
            output_dir_for_patches_of_slide (str): The path to the output directory for patches of this slide.
        """
        # check if there are as many files in directory as entries in the csv file; filter out non pt files
        patches_pt_files = os.listdir(output_dir_for_patches_of_slide)
        patches_pt_files = [filename for filename in patches_pt_files if filename.endswith('.pt')]
        if len(patches_pt_files) != len(metadata['filename']):
            logging.error(f'Number of patches ({len(patches_pt_files)}) and number of metadata entries ({len(metadata["filename"])}) do not match in directory {output_dir_for_patches_of_slide}.')
            raise Exception(f'Number of patches ({len(patches_pt_files)}) and number of metadata entries ({len(metadata["filename"])}) do not match in directory {output_dir_for_patches_of_slide}. Have you sampled patches from this slide mutliple times?')

        # transform metadata to a pandas data frame and save it to file
        df_metadata = pd.DataFrame(metadata)
        filename_csv = os.path.join(output_dir_for_patches_of_slide, 'metadata.csv')
        df_metadata.to_csv(filename_csv, index=False) 
        # log some information
        logging.info(f'Metadata of this slide saved to "{filename_csv}"')

    def switch_string_to_int_label(self, metadata_of_one_slide: dict) -> dict:
        """
        Switch the string label to an int label.

        Args:
            metadata_of_one_slide (dict): Dict containing the metadata of one slide.

        Returns:
            dict: Updated metadata dict.
        """
        # get the diagnosis from the metadata dict
        diagnosis = metadata_of_one_slide['diagnosis']
        # deal with cases where diagnosis is not a string or not in the labels_map_string_to_int
        try:
            # ensure that the diagnosis is in all capital letters
            diagnosis = diagnosis.upper()
            metadata_of_one_slide['diagnosis'] = self.labels_map_string_to_int[diagnosis]
        except:
            logging.warning(f"Could either not convert diagnosis {diagnosis} to all capital letters or diagnosis is not part of labels_map_string_to_int. Assigning label 0, i.e. 'Unknown'")
            metadata_of_one_slide['diagnosis'] =  self.labels_map_string_to_int['Unknown']    


    def get_parameters_of_slide(self, path: str) -> tuple:
        """
        Get the parameters of a slide from the database.

        Args:
            path (str): The path to the slide.

        Returns:
            tuple: A tuple containing the coordinates and center points of the patches that will be sampled from the slide, the maximal level of the slide, the patch size in pixels the metadata of the slide and the size of a tile in pixels.
        """
        # get the maximal level of the slide needed to get the coordinates from the database
        max_level = self.get_max_level_from_database(path)
        # if the patch size is larger than the size of the tiles at the highest zoom level, we have to choose the next lower level
        # note that lower levels represent bigger patches
        corresponding_level = self.compute_level_corresponding_to_patch_size(max_level)
        if self.general_transform_config['sampling']['highest_zoom_level'] is True:
            level_to_sample_from = max_level
        else:
            level_to_sample_from = corresponding_level
        # the original coordinates are needed to precompute the coordinates for sampling
        coords_of_slide = self.get_coordinates_from_database(path, corresponding_level)
        # precompute the coordinates to sample from
        coords_for_sampling = self.get_relevant_tile_coordinates(coords_of_slide, path, corresponding_level, 
                                                                 level_to_sample_from, self.output_config['white_threshold'])
        metadata_of_one_slide = self.get_metadata_for_one_slide(path) 
        return coords_for_sampling, level_to_sample_from, metadata_of_one_slide

    #########################################################################################################################################
    ############################################ Utility functions for get_parameters_of_slide() ############################################
    #########################################################################################################################################

    def get_max_level_from_database(self, path: str) -> int:
        """ Retrieve the maximal zoom level from the 'tiles' table in a SQLite database. """
        # get the maximum level from the tiles table if possible
        try: 
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()
                # take the maximal zoom level from the database
                cursor.execute("SELECT MAX(level) FROM tiles")  
                max_level = cursor.fetchone()[0]
                cursor.close()
                logging.debug(f'max_level: {max_level}')
                return max_level
        except:
            logging.error("Error occurred while accessing tiles table for max level")
            raise Exception("Error occurred while accessing tiles table for max level")
        
    def compute_level_corresponding_to_patch_size(self, max_level) -> int:
        """ Compute the level that corresponds to the patch size and the level to sample from, i.e. patch size 512um <-> third highest level. """
        if max_level is None:
            logging.warning("Max level could not be found, returning None")
            return None
        corresponding_level = int(np.floor(max_level - np.log2(self.patch_size_in_pixels/self.tile_size)))
        logging.debug(f'corresponding_level: {corresponding_level}')

        if corresponding_level < 0 or corresponding_level > max_level:
            logging.warning(f'Corresponding level is smaller than 0 or greater than max level, returning None')
            return None
        return corresponding_level
        

    def get_coordinates_from_database(self, path: str, corresponding_level: int) -> tuple:
        """
        Retrieve all coordinates from the 'tiles' table in a SQLite database.
            
            Args: 
                path (str): The path to the SQLite database.
                corresponding_level (int): The zoom level from which the tiles should be sampled.
            
            Returns: 
                Tuple: A tuple containing all coordinates and the max level.
        """
        if corresponding_level is None:
            logging.warning("Corresponding level could not be found, returning None")
            return None
        # get the coordinates from the tiles table if possible
        try:
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()
                # take all coordinates from max zoom level from the database
                cursor.execute("SELECT x, y FROM tiles WHERE level=?", (corresponding_level,))  
                coords = cursor.fetchall()
                cursor.close()
        except:
            logging.error("Error occurred while accessing tiles table for coordinates")
            raise Exception("Error occurred while accessing tiles table for coordinates")
            
        # check if there are any coordinates in the database
        if not coords:
            logging.warning("Set of coordinates found in tiles table is empty")
            return None
        logging.debug(f'coords_of_slide {coords} at corresponding level {corresponding_level}')
        return coords
    
    def get_resolutions_from_database(self, path: str) -> tuple[int, int]:
        """
        Retrieve the resolution values from the 'metadata' table of a SQLite database.
        
            Args: 
                path (str): The path to the SQLite database.

            Returns: 
                Tuple[int, int]: A tuple containing the resolution values as integers.
        """
        # get the resolution values from the metadata table if possible
        try:
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                        SELECT value 
                        FROM metadata 
                        WHERE key LIKE 'resolution%ppm' 
                            OR key LIKE 'x_ppm' 
                            OR key LIKE 'y_ppm'
                """)
                resolutions = cursor.fetchall()
                cursor.close()
        except:
            resolutions = [] # since this is the output if resolutions is not in the metadata table
            logging.warning("No resolution values found in metadata table") # also raise exception?
        
        # check if there are any resolution values in the database and if not, use the default values given in config.yaml
        if resolutions == []:
            resolutions = [self.general_transform_config['sampling']['wsi_pixels_per_m'], 
                           self.general_transform_config['sampling']['wsi_pixels_per_m']]
        else:
            # Convert the resolution values to integers, always take the first two values of the resolution list
            resolutions = [int(resolutions[0][0]), int(resolutions[1][0])]
        return tuple(resolutions)
    

    def get_tile_size_from_database(self, path: str) -> int:
        """ Retrieve the tile size from the 'metadata' table in a SQLite database. 
        
        Args:
            path (str): The path to the SQLite database.
            
        Returns:
            int: The tile size.
        """
        # get the tile size from the metadata table if possible
        try:
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT value FROM metadata WHERE key='tile_size'")
                tile_size = cursor.fetchone()
                cursor.close()
        except:
            logging.warning("Error occurred while accessing metadata table for tile size, using default value of 512")
            tile_size = None
            
        return int(tile_size[0]) if tile_size is not None else 512


    ####################################################################################################################################

    def get_metadata_for_one_slide(self, path: str) -> list: 
        """
        Retrieve the metadata from the 'metadata' table in a SQLite database.

        Args:
            path (str): The path to the SQLite database.

        Returns:
            list: A list containing the metadata of the form [('filename', filename), ('diagnosis', diagnosis),...].
        """
        # setup container for metadata
        metadata_of_one_slide = {attributes : None for attributes in self.output_config['desired_metadata']}
        # get the metadate from current slide 
        metadata_of_one_slide = self.get_metadata_from_sqlite_file(path, metadata_of_one_slide)
        logging.debug(f'metadata_of_one_slide: {metadata_of_one_slide}')
        return metadata_of_one_slide
    
    #################################### Utility function for get_metadata_for_one_slide() #############################################

    def get_metadata_from_sqlite_file(self, path: str, metadata: dict) -> dict:
        """
        Get metadata from sqlite file and save to a dict.

        Args:
            path (str): path to sqlite file
            metadata (dict): dict to store metadata

        Returns:
            dict: dict with metadata of one slide
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
                    metadata[key] = row[0][0] 
                else:
                    logging.debug(f"No {key} found in metadata table")
                    metadata[key] = None 
        return metadata
    
    ####################################################################################################################################

    def finish_preprocessing_patch(self, output_dir_for_patches_of_slide: str, metadata_of_one_slide: dict, coordinates: list, metadata: list, one_patch_np: np.ndarray, label_for_slide: torch.tensor):
        """
        Finish preprocessing one patch by augmenting it, appending the metadata to the metadata dict and save the patch to file.

        Args:
            output_dir_for_patches_of_slide (str): The path to the output directory for patches of this slide.
            metadata_of_one_patch (dict): Dict containing the metadata of the patch.
            coordinates (list): A list containing the coordinates of the patch.
            metadata (list): A list containing the metadata of all patches from all slides.
            one_patch_np (np.ndarray): The preprocessed patch.
            label_for_slide (torch.tensor): The label of the slide.
        """
        # remove whitespace from coordinates to use them in the filename
        coords_as_string = str(coordinates)
        coords_as_string = coords_as_string.replace(' ', '')
        path_to_patch = os.path.join(output_dir_for_patches_of_slide, f'patch_{self.processed_patches}_coords_{coords_as_string}.pt')
        # append the corresponding coordinate to the metadata of the current slide
        metadata_of_this_patch = metadata_of_one_slide.copy()
        # add the coordinates of the patch and the filename to the metadata dict
        metadata_of_this_patch['coordinates'] = coordinates
        metadata_of_this_patch['filename'] = os.path.basename(path_to_patch)
        # append the values for this patch to the metadata dict -> will have length (number of patches for this slide)
        metadata = self.append_values_to_metadata(metadata, metadata_of_this_patch)
        if self.mode == 'train':
            augmented_patch = augment.augment_tile(one_patch_np, self.training_transform_config)
        elif self.mode == 'test':
            # simply switch to tensor
            augmented_patch = augment.to_torch_tensor(one_patch_np)
        else:
            logging.error(f'Unknown mode {self.mode}')
            raise Exception(f'Unknown mode {self.mode}')
        # save the patch and its label and update the progress 
        torch.save((augmented_patch, label_for_slide), path_to_patch)
    
    def append_values_to_metadata(self, metadata: dict, metadata_of_this_patch: dict) -> dict:
        """
        Append the metadata of one patch to the metadata of all patches from one slide.

        Args:
            metadata (dict): Dict containing the metadata that should be stored.
            metadata_of_this_patch (dict): Dict containing the metadata of one patch.

        Returns:
            dict: Updated metadata dict.
        """
        for key in metadata.keys():
            metadata[key].append(metadata_of_this_patch[key])
        return metadata

    ############################################ Utilities for get_parameters_of_slide() #############################################
    
    def get_relevant_tile_coordinates(self, original_coordinates, path: str, corresponding_level: int, 
                                      level_to_sample_from: int, white_threshold: float) -> list:
        """ Get the relevant tile coordinates from the original coordinates depending on the white ratio.
        
        Args:
            original_coordinates (list): A list containing the original coordinates.
            path (str): The path to the SQLite database.
            corresponding_level (int): The zoom level of the original coordinates.
            level_to_sample_from (int): The zoom level from which the tiles should be sampled.
            white_threshold (float): The white threshold.

        Returns:
            list: A list containing the relevant tile coordinates.
        """
        if original_coordinates is None or len(original_coordinates) == 0:
            logging.warning(f'Found no coordinates for this slide, returning None')
            return None
    
        if level_to_sample_from is None:
            logging.warning(f'Level to sample from could not be found, returning None')
            return None

        relevant_coordinates = []
        for coord in original_coordinates:
            # load the tile of the current coordinate
            tile = self.get_tile_from_database(coord, path, corresponding_level)
            # check if the tile has less or eqaul than r percent of white pixels
            if self.check_if_tile_is_relevant(tile, white_threshold):
                relevant_coordinates.append(coord)

        # check if we have any relevant coordinates
        if len(relevant_coordinates) == 0:
            logging.warning(f'Found no relevant tiles for this slide for white ratio less or equal than {white_threshold}, returning None')
            return None
        
        # expand the coordinates to the highest zoom level if desired by user
        if level_to_sample_from == corresponding_level:
            logging.debug(f'coords_for_sampling: {relevant_coordinates}')
            return relevant_coordinates
        else:
            coords_in_higher_zoom = self.translate_coordinates_to_highest_zoom_level(relevant_coordinates, corresponding_level, level_to_sample_from)
            logging.debug(f'coords_for_sampling: {coords_in_higher_zoom}')
            return coords_in_higher_zoom   

    
    
    def get_tile_from_database(self, coord: tuple, path: str, corresponding_level: int) -> np.ndarray:
        """ Get the tile from the database. 
        
        Args:
            coord (tuple): The coordinates of the tile.
            path (str): The path to the SQLite database.
            corresponding_level (int): The zoom level of the tile.

        Returns:
            np.ndarray: The tile as a NumPy array.
        """
        x, y = coord
        try:
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT jpeg FROM tiles WHERE x=? AND y=? AND level=?", (x, y, corresponding_level))
                row = cursor.fetchone()
                if row is not None:
                    jpeg_data = row[0]
                    with BytesIO(jpeg_data) as bytes_io:
                        with Image.open(bytes_io) as pil_image:
                            img = np.array(pil_image)
                else:
                    logging.warning(f'No tile found for coordinate {coord} at level {corresponding_level}')
                    img = None
                cursor.close()
        except:
            logging.error("Error occurred while connecting to sqlite database")
            raise Exception("Error occurred while connecting to sqlite database")
        return img
    
    def check_if_tile_is_relevant(self, tile: np.ndarray, white_threshold: float) -> bool:
        """ Check if the tile is relevant, i.e. has less or equal than r white pixels.

        Args:
            tile (np.ndarray): The tile as a NumPy array.
            white_threshold (float): The white threshold.

        Returns:
            bool: True if the tile is relevant, False otherwise.
        """
        if tile is None:
            return False
        # check if the tile has less or eqal than r white pixels
        if np.mean(tile == 255) >= white_threshold:
            return False
        
        return True
    
    def translate_coordinates_to_highest_zoom_level(self, relevant_coordinates: list, corresponding_level: int, level_to_sample_from: int) -> list:
        """ Translate the coordinates to the highest zoom level. 
        
        Args:
            relevant_coordinates (list): A list containing the relevant coordinates.
            corresponding_level (int): The zoom level of the original coordinates.
            level_to_sample_from (int): The zoom level from which the tiles should be sampled.

        Returns:
            list: A list containing the coordinates at the highest zoom level.
        """
        diff = level_to_sample_from - corresponding_level
        # obtain the factor by which the coordinates have to be multiplied to get to the highest zoom level
        level_multiplier = 2**(diff) # this is also the number of tiles in x and y direction
        coords_in_higher_zoom = []

        if self.general_transform_config['sampling']['overlap_bool'] == False or diff == 0:
            for coord in relevant_coordinates:
                coords_in_higher_zoom.append([(coord[0]*level_multiplier + i, 
                                               coord[1]*level_multiplier + j) 
                                               for i in range(level_multiplier) for j in range(level_multiplier)])
                
            if len(coords_in_higher_zoom) != len(relevant_coordinates):
                logging.error(f"Error occurred while computing coordinates for higher zoom level")
                raise Exception("Error occurred while computing coordinates for higher zoom level")  
                                                
        else: 
            for coord in relevant_coordinates: 
                # also append the coordinates of the overlapping patches
                for l, m in [(0, 0), (0, 1), (1, 0), (1, 1)]:
                    coords_in_higher_zoom.append([(coord[0]*level_multiplier + i + l*(level_multiplier // 2), 
                                                   coord[1]*level_multiplier + j + m*(level_multiplier // 2) ) 
                                                   for i in range(level_multiplier) for j in range(level_multiplier)])

            if len(coords_in_higher_zoom) != 4 * len(relevant_coordinates):
                logging.error(f"Error occurred while computing overlaping coordinates for higher zoom level")
                raise Exception("Error occurred while computing overlaping coordinates for higher zoom level") 
 
        
        return coords_in_higher_zoom


    ###################################################################################################################################

    def get_patch_size_in_pixels(self, resolutions: tuple) -> int:
        """ Calculate the patch size in pixels based on the resolution values. Note that in order to ensure
            compatibility with already sampled patches, the patch size is cut to a square defined by the resolution
            of the already sampled patches.
            
            Args: 
                resolutions (Tuple[int, int]): A tuple containing the resolution values as integers.
            
            Returns: 
                int: The size of the patch in pixels.
        """
        # check if patch size corresponds to a size resulting from the resolution values

        # Calculate the patch size in pixels based on the resolution values
        patch_size_in_meters = 1e-6 * self.general_transform_config['sampling']['patch_size_um']
        if resolutions[0] != self.general_transform_config['sampling']['wsi_pixels_per_m'] or resolutions[1] != self.general_transform_config['sampling']['wsi_pixels_per_m']:
            default_resolution = self.general_transform_config['sampling']['wsi_pixels_per_m']
            logging.warning(f'Resolution values {resolutions} do not match the default value: {default_resolution}, using the default value anyway to ensure compatibility with already sampled patches.')
            x_pixels = int(self.general_transform_config['sampling']['wsi_pixels_per_m'] * patch_size_in_meters)
            y_pixels = x_pixels
            logging.info(f"The patches correspond to a size of {x_pixels/(resolutions[0]*1e-6)}x{x_pixels/(resolutions[1]*1e-6)} um instead of {self.general_transform_config['sampling']['patch_size_um']} um")
        else:
            x_pixels = int(resolutions[0] * patch_size_in_meters)
            y_pixels = int(resolutions[1] * patch_size_in_meters)

        # Cut the patch size to a square defined by the lower resolution
        if x_pixels <= y_pixels:
            y_pixels = x_pixels
        else:
            x_pixels = y_pixels
        # check if we get a positive patch size
        if x_pixels <= 0:
            logging.warning(f'Patch size in pixels is {x_pixels},  raising exception')
            raise Exception(f'Patch size in pixels is negative: {x_pixels}')
        if np.log2(x_pixels/self.tile_size) % 1 != 0:
            logging.warning(f'Patch size in pixels = {x_pixels} is not a multiple of the tile size (also in pixels) = {self.tile_size}, raising exception')
            raise Exception(f'Patch size in pixels = {x_pixels} is not a multiple of the tile size (also in pixels) = {self.tile_size}')
        return x_pixels
    
############################################################################################################################


def main(args):
    """ Main function for preprocessing.

    Args:
        args (argparse.Namespace): The arguments passed to the script.  
    """
    # to avoid circular imports
    from patchcraft.sample_tiles.config import Config             # import configuration class

    # setup config and a directory for all the output
    c = Config(args)
    c = c.setup_config()

    # set random seed for numpy.random and for torch, default (=0) is the datetime 
    seed = c['general_transforms']['sampling']['random_seed']
    if seed == 0:
        # set seed
        seed = c['general_transforms']['sampling']['patch_size_um']
        c['general_transforms']['sampling']['random_seed'] = seed # such that its saved in the config file

    torch.manual_seed(seed)
    random.seed(seed)
    
    sampler = GenerateData(config=c)

    # do the preprocessing
    print("Start timing")
    start = time.time()
    sampler.preprocess_patches()
    end = time.time()
    print()
    print("Time elapsed: ", end - start)

    # plot some information about the first slide
    slide = os.listdir(c['output']['path'])[0]
    path = os.path.join(c['output']['path'], slide)
    print("Some information about the first slide:")
    df_metadata = pd.read_csv(os.path.join(path, 'metadata.csv'))
    print("df_metadata: \n", df_metadata)
    patches = os.listdir(path)
    patches = [patch for patch in patches if patch.endswith('.pt')]
    print("Number of patches: ", len(patches))
    


