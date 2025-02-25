import os                                             # for file and directory operations
import logging                                        # for logging errors, warnings, and info messages
import torch          
import numpy as np  
import pandas as pd
import time
import sqlite3
import math
import yaml
from PIL import Image
from io import BytesIO
from numpy import random 

from patchcraft.diagnosis_maps import LABELS_MAP_STRING_TO_INT

import patchcraft.sample_tiles.augment as augment           # needed for to_torch_tensor and resize functions
import patchcraft.utils as utils                            # import utility functions for logging and progress bar



class GenerateData(): 
    """ Class for preprocessing patches from slides. """
    def __init__(self, config):
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
        self.overlap = self.general_transform_config['sampling']['overlap']
        self.mode = self.output_config['mode'] 
        
        # setup labels for the different classes -> will be used for switching from string to int labels
        self.labels_map_string_to_int = LABELS_MAP_STRING_TO_INT
        

    def preprocess_one_patch(self, path: str, current_coords: list, level_to_sample_from: int, patch_size_in_pixels: int, tile_size: int) -> np.ndarray:
        """
        Preprocess one patch from a slide.

        Args:
            path (str): The path to the slide.
            current_coords (list): A list containing the coordinates (from sqlite database) and the center point of the patch (coordinates that point to a point within a tile).
            level_to_sample_from (int): The zoom level from which the tiles should be sampled.
            patch_size_in_pixels (int): The size of the patch in pixels.
            tile_size (int): The size of the tiles in pixels.

        Returns:
            numpy.ndarray: The preprocessed patch.
        """
        tile_coords, patch_coords = current_coords
        logging.debug(f'tile coords: {tile_coords}, patch_coords: {patch_coords}')
        one_patch_np = self.get_tiles_and_combine_to_patch(path, patch_size_in_pixels, tile_coords, patch_coords, level_to_sample_from, tile_size) 
        # if there was at least one tile missing, None was returned
        if one_patch_np is None: 
            return None
        logging.debug(f'one_patch_np.shape: {one_patch_np.shape}')
        return one_patch_np
    
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
        coords_for_sampling, level_to_sample_from, patch_size_in_pixels, metadata_of_one_slide, tile_size = self.get_parameters_of_slide(path_to_slide) 
        # if there was a problem with the parameters, None was returned
        if coords_for_sampling is None or level_to_sample_from is None or patch_size_in_pixels is None or metadata_of_one_slide is None or tile_size is None:
            logging.warning(f'Some parameters for slide {path_to_slide} could not be found, returning None')
            return None 
        
        # switch from string to int labels, get the length of coords_for_sampling and the label for the slide
        self.switch_string_to_int_label(metadata_of_one_slide) # i.e. metadata_of_one_slide['diagnosis'] -> int
        label_for_slide = metadata_of_one_slide['diagnosis']
        label_for_slide = torch.tensor(label_for_slide)

        # iterate through all coordinates = (tile coordinates, patch coordinates) and preprocess one patch at a time
        len_coords_for_sampling = len(coords_for_sampling) if self.output_config['number_of_patches_per_slide'] == 'all' else min(len(coords_for_sampling), int(self.output_config['number_of_patches_per_slide']))
        sampling_possible = False
        for coords in coords_for_sampling: # coords are (tile_coords, patch_coords)
            # break if we have sampled the desired number of patches
            if self.output_config['number_of_patches_per_slide'] != 'all' and self.processed_patches >= int(self.output_config['number_of_patches_per_slide']):
                logging.info(f'Number of processed patches ({self.processed_patches}) reached the number of patches to be sampled ({self.output_config["number_of_patches_per_slide"]}) for slide {path_to_slide}, continuing with next slide')
                print(f"Number of processed patches ({self.processed_patches}) reached the number of patches to be sampled ({self.output_config['number_of_patches_per_slide']}) for slide {path_to_slide}, continuing with next slide")
                break
            for _ in range(self.output_config['number_of_repeated_patches']):
                # scale patch size and perturbe coordinates depending on the mode
                scaled_patch_size_in_pixels, coords = self.scale_patch_size_and_perturbe_coordinates(patch_size_in_pixels, coords, tile_size)
                # preprocess one patch if possible 
                one_patch_np = self.preprocess_one_patch(path_to_slide, coords, level_to_sample_from, scaled_patch_size_in_pixels, tile_size)
                if isinstance(one_patch_np, np.ndarray):
                    # augment patch if in training mode and save it to file
                    self.finish_preprocessing_patch(output_dir_for_patches_of_slide, metadata_of_one_slide, coords, metadata, one_patch_np, label_for_slide)
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
        """ Preprocess patches from multiple slides and saves them as single files. """
        # get all slide names from the input directory
        slide_names = os.listdir(self.input_config['path'])
        # select n slides with correct stain starting at start_slide (n=number of slides)
        stain = self.general_transform_config['sampling']['stain']
        slide_names = [filename for filename in slide_names if filename.endswith('.sqlite') and stain in filename]
        slide_names.sort() # for reproducibility
        slide_names = slide_names[self.output_config['start_slide']:] # count from 0
        if self.output_config['number_of_slides'] != 'all':
            slide_names = slide_names[:int(self.output_config['number_of_slides'])]
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
                logging.error(f"Output directory {output_dir_for_patches_of_slide} already exists")
                raise Exception(f"Output directory {output_dir_for_patches_of_slide} already exists")
            else:
                os.makedirs(output_dir_for_patches_of_slide)

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

    def scale_patch_size_and_perturbe_coordinates(self, patch_size_in_pixels: int, coords: list, tile_size: int) -> tuple:
        """ Scale patch size and perturbe coordinates depending on the mode
         
        Args:
            patch_size_in_pixels (int): Size of the patch in pixels.
            coords (list): A list containing the tile coordinates (from sqlite database) and the patch coordinates (point to a point within a tile).
            tile_size (int): The size of the tiles in pixels.

        Returns:
            tuple: A tuple containing the scaled patch size in pixels and the perturbed coordinates.
        """
        if self.mode == 'train':
            # randomly scale patch size in pixels (this is the size pre rotation)
            scaled_patch_size_in_pixels = self.randomly_scale_patch_size_in_pixels_pre_rotation(patch_size_in_pixels)
            # randomly perturbe the patch coordinates (i.e the point within a tile not the coordinates of the tiles)
            coords = self.perturbe_patch_coordinates(coords, tile_size)
        elif self.mode == 'test':
            scaled_patch_size_in_pixels = patch_size_in_pixels
        else:
            logging.error(f'Unknown mode {self.mode}')
            raise Exception(f'Unknown mode {self.mode}')
        return scaled_patch_size_in_pixels, coords

    def perturbe_patch_coordinates(self, coords: list, tile_size: int) -> list:
        """ Perturbe the patch coordinates by a random amount. 
        
        Args:
            coords (list): A list containing the tile coordinates (from sqlite database) and the patch coordinates (point to a point within a tile).
            tile_size (int): The size of the tiles in pixels.

        Returns:
            list: The perturbed coordinates.
        """
        if self.training_transform_config['sampling']['perturbation_range'] <= 0:
            return coords
        perturbation_range = self.training_transform_config['sampling']['perturbation_range'] * tile_size
        random_perturbation = np.random.randint(-perturbation_range, perturbation_range, size=2)
        coords[1][0] = (coords[1][0] + random_perturbation[0]) % tile_size
        coords[1][1] = (coords[1][1] + random_perturbation[1]) % tile_size
        return coords

    
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

    def get_tiles_and_combine_to_patch(self, path: str, patch_size_in_pixels: int, tile_coords: tuple[int, int], patch_coords: tuple[int, int], level_to_sample_from: int, tile_size: int) -> np.ndarray:
        """
        Retrieve tiles from the database and combine them into a patch.

        Args:
            path (str): Path to the SQLite database.
            patch_size_in_pixels (int): Size of the patch in pixels.
            tile_coords (tuple): Random coordinates.
            patch_coords (tuple): Coordinates of the center point of the patch.
            level_to_sample_from (int): Zoom level from which the tiles should be sampled.
            tile_size (int): Size of one tile in pixels.

        Returns:
            numpy.ndarray: Combined tiles as a patch.
        """
        x_plus, x_minus, y_plus, y_minus = self.compute_number_of_extra_tiles(patch_size_in_pixels, patch_coords, tile_size)
        logging.debug(f'x_plus: {x_plus}, x_minus: {x_minus}, y_plus: {y_plus}, y_minus: {y_minus}')
        # Define the starting point for iterating through the necessary tiles & initialize data container for the tiles of the patch
        starting_point = [tile_coords[0] - x_minus, tile_coords[1] - y_minus]
        logging.debug(f'starting_point for iterating through needed tiles: {starting_point}')
        tiles_for_one_patch = []
        try:
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()
        except:
            logging.error("Error occurred while connecting to sqlite database")
            raise Exception("Error occurred while connecting to sqlite database")
        else:
            for xi in range(starting_point[0], starting_point[0] + x_plus + x_minus + 1):
                y_tiles = self.get_y_tiles_for_one_x(cursor, xi, starting_point[1], y_plus, y_minus, level_to_sample_from)
                if not y_tiles:
                    return None
                # Concatenate the tiles in the y-direction
                y_tiles = np.concatenate(y_tiles, axis=0) 
                tiles_for_one_patch.append(y_tiles)
            cursor.close()

        # Concatenate the tiles in the x direction
        tiles_for_one_patch = np.concatenate(tiles_for_one_patch, axis=1) 
        # Note: numpy dimensions (matrix) neq x,y dimensions (coordinates) -> switch axis 0 and 1 before slicing
        tiles_for_one_patch = np.transpose(tiles_for_one_patch, (1,0,2))
        logging.debug(f'tiles_for_one_patch.shape before: {tiles_for_one_patch.shape}')
        # get the wanted patch of wanted size
        tiles_for_one_patch = self.delete_border_sections(tiles_for_one_patch, patch_size_in_pixels, patch_coords, tile_size, x_minus, y_minus)
        # it is possible that all tiles are white since we cut the patch to the desired size 
        if np.all(tiles_for_one_patch == 255):
            return None
        logging.debug(f'tiles_for_one_patch.shape after:  {tiles_for_one_patch.shape}')        
        return tiles_for_one_patch
    

    #################################### Utility functions for get_tiles_and_combine_to_patch() ##########################################

    def compute_number_of_extra_tiles(self, patch_size_in_pixels: int, patch_coords: tuple[int, int], tile_size: int) -> tuple:
        """
        Compute the number of extra tiles needed to create a patch of the desired size.

        Args:
            patch_size_in_pixels (int): Size of the patch in pixels.
            patch_coords (tuple): Coordinates of the center point of the patch.
            tile_size (int): Size of one tile in pixels.
        Returns:
            tuple: A tuple containing the number of extra tiles in the x and y direction.
        """
        half_size = patch_size_in_pixels/2
        x_plus = math.ceil((half_size + patch_coords[0]) / tile_size - 1) # -1 since starting tile is excluded
        x_minus = math.ceil((half_size - patch_coords[0]) / tile_size) 
        y_plus = math.ceil((half_size + patch_coords[1]) / tile_size - 1)
        y_minus = math.ceil((half_size - patch_coords[1]) / tile_size)
        return x_plus, x_minus, y_plus, y_minus

    def get_y_tiles_for_one_x(self, cursor: sqlite3.Cursor, xi: int, starting_y: int, y_plus: int, y_minus: int, level_to_sample_from: int) -> list:
        """
        Retrieve tiles from the database for one x-coordinate and combine them into a list.

        Args:
            cursor (sqlite3.Cursor): Cursor to execute SQL queries.
            xi (int): current x-coordinate.
            starting_y (int): starting y-coordinate.
            y_plus (int): number of tiles in positive y-direction (starting tile is excluded).
            y_minus (int): number of tiles in negative y-direction (starting tile is excluded).
            level_to_sample_from (int): Zoom level from which the tiles should be sampled.

        Returns:
            list: A list containing the tiles in the y-direction.
        """
        y_tiles = []
        # Iterate through the necessary tiles in the y-direction
        for yi in range(starting_y, starting_y + y_plus + y_minus + 1):
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
                return []  
        return y_tiles
    

    def delete_border_sections(self, tiles_for_one_patch: np.ndarray, patch_size_in_pixels: int, center_point_of_patch: tuple[int, int], tile_size: int, x_minus: int, y_minus: int) -> np.ndarray:
        """
        Delete the border sections of the tiles to get the desired patch size.

        Args:
            tiles_for_one_patch (numpy.ndarray): Tiles of the patch.
            patch_size_in_pixels (int): Size of the patch in pixels.
            center_point_of_patch (tuple): Coordinates of the center point of the patch.
            tile_size (int): Size of one tile in pixels.
            x_minus (int): number of tiles in negative x-direction (starting tile is excluded).
            y_minus (int): number of tiles in negative y-direction (starting tile is excluded).

        Returns:
            numpy.ndarray: Tiles of the patch with the desired size.
        """
        # Compute the center from the wanted patch
        center_of_patch = [x_minus*tile_size + center_point_of_patch[0], y_minus*tile_size + center_point_of_patch[1]]
        half_size = patch_size_in_pixels/2
        if center_of_patch < [half_size, half_size]:
            logging.error("center_of_patch: ", center_of_patch)
            logging.error("half_size: ", half_size) 
            logging.error("Location of 'center_of_patch' would lead to negative indices")
            raise Exception("Location of 'center_of_patch' would lead to negative indices")
        logging.debug(f'center_of_patch: {center_of_patch}')
        # Get the x,y indices for correctly slicing the array of tiles and slice it
        patch_x_start = math.ceil(center_of_patch[0] - half_size)
        patch_x_end = math.ceil(center_of_patch[0] + half_size)
        patch_y_start = math.ceil(center_of_patch[1] - half_size)
        patch_y_end = math.ceil(center_of_patch[1] + half_size)
        logging.debug(f'patch_x_start: {patch_x_start}, patch_x_end: {patch_x_end}, patch_y_start: {patch_y_start}, patch_y_end: {patch_y_end}')
        one_patch = tiles_for_one_patch[patch_x_start : patch_x_end, patch_y_start : patch_y_end]
        return one_patch
    
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

    def switch_string_to_int_label(self, metadata_of_one_slide: dict):
        """
        Switch the string label to an int label.

        Args:
            metadata_of_one_slide (dict): Dict containing the metadata of one slide.
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
        # the resolutions are needed to calculate the patch size, if not there we set it to default value
        resolutions_of_slide = self.get_resolutions_from_database(path)
        logging.debug(f'resolutions_of_slide: {resolutions_of_slide}')
        # note that on every level the tile size in pixels is the same BUT they correspond to different physical sizes!
        tile_size = self.get_tile_size_from_database(path) # will also be set to default value if not in database
        logging.debug(f'tile_size: {tile_size}')
        patch_size_in_pixels = self.get_patch_size_in_pixels(resolutions_of_slide) # raises exception if patch size is negative
        logging.debug(f'patch_size_in_pixels: {patch_size_in_pixels}')
        # get the maximal level of the slide -> needed to get the coordinates from the database
        max_level = self.get_max_level_from_database(path)
        logging.debug(f'max_level: {max_level}')
        if max_level is None:
            logging.warning(f'No maximum level found in tiles table for slide {path}')
            return None, None, None, None, None
        if self.general_transform_config['sampling']['highest_zoom_level'] == True:
            level_to_sample_from = max_level
        else:
            level_to_sample_from = self.get_next_higher_level_from_database(tile_size, patch_size_in_pixels, max_level)
            # scale the patch size in pixels to the corresponding level such that we have everything relative to the coordinates of the sqlite and the tile size of 512px
            patch_size_in_pixels = patch_size_in_pixels * 2**(-(max_level - level_to_sample_from))
        logging.debug(f'level_to_sample_from: {level_to_sample_from}')
        if level_to_sample_from is None:
            logging.warning(f'No next higher level found for slide {path}')
            return None, None, None, None, None
        # the original coordinates are needed to precompute the coordinates for sampling
        coords_of_slide = self.get_coordinates_from_database(path, level_to_sample_from)
        logging.debug(f'coords_of_slide: {coords_of_slide}')
        if coords_of_slide is None:
            return None, None, None, None, None
        # precompute the coordinates to sample from
        coords_for_sampling = self.precompute_tile_and_patch_points(coords_of_slide, tile_size, patch_size_in_pixels)
        if coords_for_sampling is None:
            return None, None, None, None, None
        logging.debug(f'coords_for_sampling: {coords_for_sampling}')
        metadata_of_one_slide = self.get_metadata_for_one_slide(path) # not bad if this is None -> not needed to execute the rest of the code
        logging.debug(f'metadata_of_one_slide: {metadata_of_one_slide}')
        return coords_for_sampling, level_to_sample_from, patch_size_in_pixels, metadata_of_one_slide, tile_size

    #########################################################################################################################################
    ############################################ Utility functions for get_parameters_of_slide() ############################################
    #########################################################################################################################################

    def get_max_level_from_database(self, path: str) -> int:
        """
        Retrieve the maximal zoom level from the 'tiles' table in a SQLite database.

        Args:
            path (str): The path to the SQLite database.

        Returns:
            int: The maximal zoom level.
        """
        # get the maximum level from the tiles table if possible
        try: 
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()
                # take the maximal zoom level from the database
                cursor.execute("SELECT MAX(level) FROM tiles")  
                max_level = cursor.fetchone()[0]
                cursor.close()
                return max_level
        except:
            logging.error("Error occurred while accessing tiles table for max level")
            raise Exception("Error occurred while accessing tiles table for max level")
    
    def get_next_higher_level_from_database(self, tile_size: int, patch_size_in_pixels: int, max_level: int) -> int:
        """
        Retrieve the next higher zoom level from the 'tiles' table in a SQLite database.

        Args:
            tile_size (int): The size of the tiles in pixels.
            patch_size_in_pixels (int): The size of the patch in pixels.
            max_level (int): The maximal zoom level of the slide.

        Returns:
            int: The next higher zoom level, e.g. if the tile size is 512 and the patch size in pixels is 600, the next higher zoom level is max_level-1.
        """
        for i in range(max_level):
            if tile_size * 2**i >= patch_size_in_pixels:
                return max_level - i
        return None

    def get_coordinates_from_database(self, path: str, level_to_sample_from: int) -> tuple:
        """
        Retrieve all coordinates from the 'tiles' table in a SQLite database.
            
            Args: 
                path (str): The path to the SQLite database.
                level_to_sample_from (int): The zoom level from which the tiles should be sampled.
            
            Returns: 
                Tuple: A tuple containing all coordinates and the max level.
        """
        # get the coordinates from the tiles table if possible
        try:
            with sqlite3.connect(path) as conn:
                cursor = conn.cursor()
                # take all coordinates from max zoom level from the database
                cursor.execute("SELECT x, y FROM tiles WHERE level=?", (level_to_sample_from,))  
                coords = cursor.fetchall()
                cursor.close()
        except:
            logging.error("Error occurred while accessing tiles table for coordinates")
            raise Exception("Error occurred while accessing tiles table for coordinates")
            
        # check if there are any coordinates in the database
        if not coords:
            logging.warning("Set of coordinates found in tiles table is empty")
            return None
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
                cursor.execute("SELECT value FROM metadata WHERE key LIKE 'resolution%ppm'")
                resolutions = cursor.fetchall()
                cursor.close()
        except:
            resolutions = [] # since this is the output if resolutions is not in the metadata table
            logging.warning("No resolution values found in metadata table") # also raise exception?
        
        # check if there are any resolution values in the database and if not, use the default values given in config.yaml
        if resolutions == []:
            resolutions = [self.general_transform_config['sampling']['wsi_pixels_per_m'], self.general_transform_config['sampling']['wsi_pixels_per_m']]
        else:
            # Convert the resolution values to integers
            resolutions = [int(resolutions[0][0]), int(resolutions[1][0])]
        return tuple(resolutions)
    

    def get_tile_size_from_database(self, path: str) -> int:
        """
        Retrieve the tile size from the 'metadata' table in a SQLite database.

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


    def randomly_scale_patch_size_in_pixels_pre_rotation(self, patch_size_in_pixels_pre_rotation: float) -> int:
        """
        Convert a given patch size in meters to the corresponding patch size in pixels,
        based on the resolution stored in the 'metadata' table of a SQLite database.
            
            Args:
                patch_size_in_pixels_pre_rotation (float): The size of the patch in pixels before applying any rotation.
            
            Returns: 
                int: The size of the patch in pixels.
        """
        scale_factor = random.uniform(1 - self.training_transform_config['sampling']['patch_size_range'], 1 + self.training_transform_config['sampling']['patch_size_range'])
        patch_size_in_pixels_pre_rotation = math.floor(scale_factor * patch_size_in_pixels_pre_rotation)
        # sanity check -> patch size in pixels pre rotation has to be > 0 otherwise something in code is wrong
        if patch_size_in_pixels_pre_rotation <= 0:
            logging.error(f'patch_size_in_pixels_pre_rotation = {patch_size_in_pixels_pre_rotation} <= 0')
            raise Exception(f'patch_size_in_pixels_pre_rotation = {patch_size_in_pixels_pre_rotation} <= 0')
        return patch_size_in_pixels_pre_rotation
    
    ######################################### Utilitiy function for get_patch_size_in_pixels_pre_rotation() #########################################

    def calculate_patch_size_pre_rotation(self, resolutions: tuple[int, int]) -> int:
        """
        Calculate the patch size in pixels based on the resolution values.
            
            Args: 
                resolutions (Tuple[int, int]): A tuple containing the resolution values as integers.
            
            Returns: 
                int: The size of the patch in pixels.
        """
        # Calculate the patch size in pixels based on the resolution values
        patch_size_in_meters = 1e-6 * self.general_transform_config['sampling']['patch_size_um']
        x_pixels = int(resolutions[0] * patch_size_in_meters)
        y_pixels = int(resolutions[1] * patch_size_in_meters)
        # Cut the patch size to a square defined by the lower resolution
        if x_pixels <= y_pixels:
            y_pixels = x_pixels
        else:
            x_pixels = y_pixels
        # Set patch size to sqrt_2 * x_pixels such that applying rotations don't result in final patch size < x_pixels
        patch_size_in_pixels_pre_rotation = math.ceil(math.sqrt(2) * x_pixels) # this factor will get cropped away after rotation -> see center_crop
        return patch_size_in_pixels_pre_rotation

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
            augmented_patch = augment.augment_patch(one_patch_np, self.general_transform_config, self.training_transform_config)
        elif self.mode == 'test':
            # switch to tensor and resize
            augmented_patch = augment.to_torch_tensor(one_patch_np)
            ts = self.general_transform_config['sampling']['target_size']
            augmented_patch = augment.resize(augmented_patch, target_size=[ts,ts])
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
    
    # NOTE: (x,y)_abs = (tile_x, tile_y)*512 + (patch_x, patch_y) is the absolute coord of the center of the patch
    # When (x,y)_abs is given, we can calculate the tile coordinates and the patch coordinates using that k = k//n * n + k%n
    # i.e. tile_x = x_abs // 512, tile_y = y_abs // 512, patch_x = x_abs % 512, patch_y = y_abs % 512
    def precompute_tile_and_patch_points(self, original_coordinates: list, tile_size: int, patch_size: int) -> list:
        """
        Precompute the tile and patch coordinates for sampling patches equidistantly from a slide.

        Args:
            original_coordinates (list): A list containing all coordinates.
            tile_size (int): The size of the tiles in pixels. This is the same for all levels.
            patch_size (int): The size of the patches in pixels.

        Returns:
            list: A list containing the coordinates and center points of the patches that will be sampled from the slide.
        """
        precomputed_coordinates = []
        absolute_coords = []
        # get minimum and maximum x and y tile coordinates
        x_min, x_max, y_min, y_max = self.get_min_max_coordinates(original_coordinates)
        logging.debug(f'x_min: {x_min}, x_max: {x_max}, y_min: {y_min}, y_max: {y_max}')

        # determine how many patches we can sample from the slide in the x and y direction
        distance_between_patches = patch_size * (1-self.overlap) # distance between two patches in pixels
        number_of_patches_in_x_direction = int((x_max - x_min)*tile_size / distance_between_patches) + 1 # +1 to be sure that we get all patches
        number_of_patches_in_y_direction = int((y_max - y_min)*tile_size / distance_between_patches) + 1
        logging.debug(f'number_of_patches_in_x_direction: {number_of_patches_in_x_direction}, number_of_patches_in_y_direction: {number_of_patches_in_y_direction}')
        # determine the corresponding starting and ending points in the x and y direction as absolute coordinates
        patch_coord_x, patch_coord_y = 0, 0 # initial patch coordinates
        x_start = x_min*tile_size + patch_coord_x # this is relative to the level! no scaling needed
        y_start = y_min*tile_size + patch_coord_y


        # iterate through the center points of the patches in the x-direction, starting point is defined above (see note above for better understanding)
        for x in [int(x_start + i*distance_between_patches) for i in range(number_of_patches_in_x_direction+1)]:
            # split the absolute coordinate x into tile coordinates and patch coordinates
            tile_coord_x = x // tile_size 
            patch_coord_x = x % tile_size
            # iterate through the center points of the patches in the y-direction
            for y in [int(y_start + i*distance_between_patches) for i in range(number_of_patches_in_y_direction+1)]: 
                # split the absolute coordinates into tile coordinates and patch coordinates
                tile_coord_y = y // tile_size        # using that k = k//n * n + k%n
                patch_coord_y = y % tile_size

                # add the current tile coordinates and the center point of the patch to the lists if the tile coordinates are in the original coordinates
                tile_coordinates = (tile_coord_x, tile_coord_y) # to stay consistent with the return type of get_coordinates_from_database()
                patch_coordinates = [patch_coord_x, patch_coord_y]
                if tile_coordinates in original_coordinates:
                    precomputed_coordinates.append((tile_coordinates, patch_coordinates))
                    absolute_coords.append((x,y))
                    # sanity check
                    dist_x = x - (tile_coord_x*tile_size + patch_coordinates[0])
                    dist_y = y - (tile_coord_y*tile_size + patch_coordinates[1])
                    if dist_x != 0 or dist_y != 0:
                        raise Exception("Something went wrong while translating the absolute coordinates to tile coordinates and patch coordinates")
                
        if len(precomputed_coordinates) == 0:
            logging.warning(f'Could not precompute coordinates for this slide, returning None')
            return None

        return precomputed_coordinates


    def get_min_max_coordinates(self, original_coordinates: list) -> tuple:
        """
        Get the minimum and maximum x and y coordinates from a list of coordinates.

        Args:
            original_coordinates (list): A list containing all coordinates.

        Returns:
            tuple: A tuple containing the minimum and maximum x and y coordinates.
        """
        x_coords = [coord[0] for coord in original_coordinates]
        y_coords = [coord[1] for coord in original_coordinates]
        x_min = min(x_coords)
        x_max = max(x_coords)
        y_min = min(y_coords)
        y_max = max(y_coords)
        return x_min, x_max, y_min, y_max
    
    ###################################################################################################################################

    def get_patch_size_in_pixels(self, resolutions: tuple[int, int]) -> int:
        """
        Calculate the patch size in pixels based on the resolution values.
            
            Args: resolutions (Tuple[int, int]): A tuple containing the resolution values as integers.
            
            Returns: int: The size of the patch in pixels.
        """
        # Calculate the patch size in pixels based on the resolution values
        patch_size_in_meters = 1e-6 * self.general_transform_config['sampling']['patch_size_um']
        x_pixels = int(resolutions[0] * patch_size_in_meters)
        y_pixels = int(resolutions[1] * patch_size_in_meters)
        # Cut the patch size to a square defined by the lower resolution
        if x_pixels <= y_pixels:
            y_pixels = x_pixels
        else:
            x_pixels = y_pixels
        # check if we get a psitive patch size
        if x_pixels <= 0:
            logging.warning(f'Patch size in pixels is {x_pixels},  raising exception')
            raise Exception(f'Patch size in pixels is negative: {x_pixels}')
        return x_pixels
    
############################################################################################################################


def main(args):
    """ Main function for preprocessing. 
    
    Args:
        args (argparse.Namespace): The parsed arguments.
    """
    # to avoid circular imports
    from patchcraft.sample_data.config import Config             # import configuration class

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
    



