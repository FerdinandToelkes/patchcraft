import argparse        # for command line arguments
import logging         # for logging
import yaml            # for config file
import os              # for checking if config file exists

def _add_arguments(parser: argparse.ArgumentParser, default_config='../config.yaml'):
    """ Add the arguments to the parser. """
    # setup path to default config file
    default_path = os.path.dirname(__file__) + '/' + default_config
    # for changing the whole config file at once: The config file contains all default values -> its name is the only default value for argparse
    parser.add_argument('-c', '--config', type=str, default=default_path, help='Path to the config file. If you choose your own config file, it has to have the same structure as the default config file, e.g. use only number_of_patches_per_blob and number_of_patches_per_slide as default values. See --help for more information.')
    # these are arguments that are found in the config file
    parser.add_argument('-p', '--path', type=str, default='.', help='Path to the directory where the output files are stored.')
    # these are only relevant for the view subcommand -> default values are not in the config file
    parser.add_argument("--index", type=int, default=0, help="The index of the first patch to plot. Index + one hundred has to be smaller than the number of patches in the blob. Default value is 0.")
    parser.add_argument("--dpi", type=int, default=200, help="The DPI of the plot. The default value is 200.")
    parser.add_argument("--output_name", type=str, default="plot.png", help="The name of the output file. The default value is 'plot.png'.")



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
        logging.info(f"Reading config file: {config_path}")
        try:
            with open(config_path, "r") as configFile:
                content = configFile.read()
                config = yaml.load(content, Loader=yaml.Loader)
        except:
            logging.error("Error occurred while opening the config file")
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
        # change output parameters of the config file if they have been given in the command line
        config = self.update_output_parameters(args, config)
        return config
    
    ######################################################################################################################################
    ########################################## Utility function for update_config_file() ################################################
    ######################################################################################################################################

    def update_output_parameters(self, args: argparse.Namespace, config: dict) -> dict:
        """
        Update the output config with the arguments given in the command line.

        Args:
            args (argparse.Namespace): Arguments given in the command line.
            config (dict): The content of the config file.

        Returns:
            dict: The updated content of the config file.
        """
        output_arguments = [
            ('path', 'path')
        ]
        for arg_name, config_key in output_arguments:
            arg_value = getattr(args, arg_name)
            if arg_value != None:
                config['output'][config_key] = arg_value
        return config
    

    ########################################################################################################################################

def main():
    pass

if __name__ == "__main__":
    # main()
    pass