import argparse        # for command line arguments
import yaml            # for config file
import os              # for checking if config file exists


def _add_arguments(parser: argparse.ArgumentParser, default_config='../config.yaml'):
    """
    Add arguments to the parser for the generate command.

    Args:
        parser (argparse.ArgumentParser): The parser.
        default_config (str, optional): The name of the default config file. Defaults to 'config.yaml'.
    """
    # setup path to default config file
    default_path = os.path.dirname(__file__) + '/' + default_config

    # for changing the whole config file at once: The config file contains all default values -> its name is the only default value for argparse
    parser.add_argument('-c', '--config', type=str, default=default_path, help='Path to the config file. If you choose your own config file, it has to have the same structure as the default config file. See --help for more information.')
    
    # for changing the input paths
    parser.add_argument('-i', '--input_path', type=str, help='Path to the directory containing the slides.')
    parser.add_argument('-if', '--info_filename', type=str, help='Path to where the info file containing the metadata for all slides is or should be saved.')

    # for changing single arguments in output config 
    parser.add_argument('-m', '--mode', type=str, help='Mode of the sampling and preprocessing. Can be either "train" or "test".')
    parser.add_argument('-o' ,'--output_path', type=str, help='Path to the directory where the sampled data should be saved.')
    parser.add_argument('-nos', '--number_of_slides', type=str, help='Number of slides that should be sampled and preprocessed. If it is set to "all", all slides will be sampled and preprocessed. This option can be used for debugging purposes.')
    parser.add_argument('--start_slide', type=int, help='Index of the first slide that should be sampled and preprocessed. This option can be used for parallel preprocessing.')
    parser.add_argument('-pps', '--number_of_patches_per_slide', type=str, help='Number of patches that should be sampled per slide. If it is set to "all", all possible patches will be sampled and preprocessed. This option can be used for debugging purposes.')
    parser.add_argument('-rp', '--number_of_repeated_patches', type=int, help='Number of repeated patches that should be sampled per slide. This option is more meaningful for the training data since augmentation is involved.')
    parser.add_argument('-wt', '--white_threshold', type=float, help='Ratio of white pixels that each sampled tile must not exceed. This option is more meaningful for the training.')
    parser.add_argument('-md', '--desired_metadata', type=list, help='List of metadata that should be saved to csv file.')
    parser.add_argument('-ll', '--log_level', help='Set the log level.')
    
    # for changing single arguments in transform config
    # sampling
    parser.add_argument('--stain', type=str, help='Stain of the slide to be sampled from.')
    parser.add_argument('-um', '--patch_size_um', type=int, help='Size of the patch in micrometers.')
    parser.add_argument('--overlap', action=argparse.BooleanOptionalAction, help="Whether to enable overlap = 0.5 between the patches. This is only used in the training mode and it's only possible when sampling >= 256um patches at the highest zoom level.")
    parser.add_argument('-res', '--wsi_pixels_per_m', type=int, help='Number of pixels per m in the WSI.')
    parser.add_argument('-hl', '--highest_zoom_level', action=argparse.BooleanOptionalAction, help='Whether to sample from the highest level of the WSI or not (pass --highest_zoom_level or --no-highest_zoom_level).')
    parser.add_argument('-r', '--random_seed', type=int, help='Random seed for better reproducibility. Default value is 0 which means that the patch size in um will be used.')
    
    # for changing single arguments in training transform config
    # flips: use BooleanOptionalAction to be able to pass --no-flips and get boolean values
    parser.add_argument('--flips', action=argparse.BooleanOptionalAction, help='Whether to randomly flip the patch or not (pass --no-flips).')
    # color jitter
    parser.add_argument('-brt', '--brightness_jitter', type=float, help='Range of the brightness jitter.')
    parser.add_argument('-con', '--contrast_jitter', type=float, help='Range of the contrast jitter.')
    parser.add_argument('-sat', '--saturation_jitter', type=float, help='Range of the saturation jitter.')
    parser.add_argument('-hue', '--hue_jitter', type=float, help='Range of the hue jitter.')
    
class Config():
    """ Class for setting up the config file. """
    def __init__(self, args: argparse.Namespace):
        """ Initialize the Config class. 
        
        Args:
            args (argparse.Namespace): Arguments given in the command line.
        """
        self.args = args
        
    def setup_config(self) -> dict:
        """ Setup the config file. 
        
        Returns:
            dict: The content of the config file.
        """
        # open the specified config file and read out its content
        config_path = self.args.config
        config = self.read_config_file(config_path)
        # update the specified config with the arguments given in the command line
        config = self.update_config_file(self.args, config)
        # create target directory for all the output files if it doesn't exist
        if not os.path.exists(config['output']['path']):
            os.mkdir(config['output']['path'])
        return config

    
    ######################################################################################################################################
    ############################################# Utility functions for setup_config() ###################################################
    ######################################################################################################################################

    def read_config_file(self, config_path: str) -> dict:
        """
        Read out the content of a config file.

        Args:
            config_path (str): Path to the config file.

        Returns:
            dict: The content of the config file.
        """
        try:
            with open(config_path, "r") as configFile:
                content = configFile.read()
                config = yaml.load(content, Loader=yaml.Loader)
        except:
            raise Exception("Error occurred while opening the config file")
        return config


    def update_config_file(self, args: argparse.Namespace, config: dict) -> dict:
        """
        Update the config file with the arguments given in the command line.

        Args:
            args (argparse.Namespace): Arguments given in the command line.
            config (dict): The content of the config file.

        Returns:
            dict: The updated content of the config file.
        """
        # change input parameters of the config file if they have been given in the command line
        config = self.update_input_parameters(args, config)
        # first update 'independent' output config such as target_size, stain etc. (they don't depend on the other input parameters)
        config = self.update_trivial_output_parameters(args, config)
        # change transform parameters of the config file if they have been given in the command line
        config = self.update_general_transform_config(args, config)
        config = self.update_training_transform_config(args, config)
        return config
    
        
    
    ######################################################################################################################################
    ########################################## Utility functions for update_config_file() ################################################
    ######################################################################################################################################

    def update_input_parameters(self, args: argparse.Namespace, config: dict) -> dict:
        """
        Update the input config with the arguments given in the command line.

        Args:
            args (argparse.Namespace): Arguments given in the command line.
            config (dict): The content of the config file.

        Returns:
            dict: The updated content of the config file.
        """
        input_arguments = [
            ('input_path', 'path'),
            ('info_filename', 'info_filename')
        ]
        for arg_name, config_key in input_arguments:
            arg_value = getattr(args, arg_name)
            if arg_value != None:
                config['input'][config_key] = arg_value
        return config


    def update_trivial_output_parameters(self, args: argparse.Namespace, config: dict) -> dict:
        """
        Update the output config with the arguments given in the command line.

        Args:
            args (argparse.Namespace): Arguments given in the command line.
            config (dict): The content of the config file.

        Returns:
            dict: The updated content of the config file.
        """
        # Define a list of output argument names and their corresponding config keys
        output_arguments = [
            ('mode', 'mode'),
            ('output_path', 'path'),
            ('number_of_slides', 'number_of_slides'),
            ('start_slide', 'start_slide'),
            ('number_of_patches_per_slide', 'number_of_patches_per_slide'),
            ('number_of_repeated_patches', 'number_of_repeated_patches'),
            ('white_threshold', 'white_threshold'),
            ('desired_metadata', 'desired_metadata'),
            ('log_level', 'log_level')
        ] 

        # change output parameters of the config file if they have been given in the command line
        for arg_name, config_key in output_arguments:
            arg_value = getattr(args, arg_name)
            if arg_value != None:
                config['output'][config_key] = arg_value
                
        return config


    def update_general_transform_config(self, args: argparse.Namespace, config: dict) -> dict:
        """
        Update the transform config with the arguments given in the command line.

        Args:
            args (argparse.Namespace): Arguments given in the command line.
            config (dict): The content of the config file.

        Returns:
            dict: The updated content of the config file.
        """
        # Define a list of transform argument names and their corresponding config keys
        transform_arguments = [
            ('stain', ['sampling', 'stain']),
            ('patch_size_um', ['sampling', 'patch_size_um']),
            ('overlap', ['sampling', 'overlap_bool']),
            ('wsi_pixels_per_m', ['sampling', 'wsi_pixels_per_m']),
            ('highest_zoom_level', ['sampling', 'highest_zoom_level']),
            ('random_seed', ['sampling', 'random_seed']),
        ] 

        # change transform parameters of the config file if they have been given in the command line
        for arg_name, config_key in transform_arguments:
            arg_value = getattr(args, arg_name)
            if arg_value != None:
                config['general_transforms'][config_key[0]][config_key[1]] = arg_value
        return config

    def update_training_transform_config(self, args: argparse.Namespace, config: dict) -> dict:
        """
        Update the transform config with the arguments given in the command line.

        Args:
            args (argparse.Namespace): Arguments given in the command line.
            config (dict): The content of the config file.

        Returns:
            dict: The updated content of the config file.
        """
        # Define a list of transform argument names and their corresponding config keys
        transform_arguments = [
            ('flips', ['flips', 'enabled']),
            ('brightness_jitter', ['color_jitter', 'brightness']),
            ('contrast_jitter', ['color_jitter', 'contrast']),
            ('saturation_jitter', ['color_jitter', 'saturation']),
            ('hue_jitter', ['color_jitter', 'hue']),
        ] 

        # change transform parameters of the config file if they have been given in the command line
        for arg_name, config_key in transform_arguments:
            arg_value = getattr(args, arg_name)
            if arg_value != None:
                config['training_transforms'][config_key[0]][config_key[1]] = arg_value
        return config

    ############################################ Utility functions ###############################################

    def get_maximal_number_of_slides_with_correct_stain(self, config: dir) -> int:
        """
        Get the maximal number of slides with correct stain from the overview file.

        Args:
            config (dir): The config dict.

        Returns:
            int: The maximal number of slides with correct stain.
        """
        # get desired stain from config
        stain = config['general_transforms']['sampling']['stain']
        # look at the overview file and compute the number of slides with correct stain
        path = config['output']['path'] + '/' + config['input']['overview_filename']
        try:
            with open(path, "r") as overview_file:
                content = overview_file.read()
                overview = yaml.load(content, Loader=yaml.Loader)
            # return the number of slides with correct stain
            return overview['stain'][stain]
        except:
            raise Exception("Error occurred while opening the overview file. Make sure that the overview file exists at the correct place and in the correct format. See --help for more information.")

    # ########################################################################################################################################

def main():
    pass

if __name__ == "__main__":
    main()
    pass
