# this structure is needed such that we can call the package directly as patchcraft instead of python -m patchcraft
def main():
    import argparse                                                                            # for command line arguments

    # Import the main functions of the subcommands
    from .create_info_file.create_info_file import main as info_main                          # for execution of the info subcommand
    from .create_info_file.config import _add_arguments as add_create_info_file_arguments      # for the argument setup for the info parser
    
    from .get_overview.get_overview import main as get_overview_main                           # for execution of the get_overview subcommand
    from .get_overview.config import _add_arguments as add_get_overview_arguments              # for the argument setup for the get_overview parser
    
    from .sample_data.sample_data import main as sample_data_main                              # for execution of the sample_data subcommand
    from .sample_data.config import _add_arguments as add_sample_data_arguments                # for the argument setup for the sample_data parser
    
    from .sample_tiles.sample_tiles import main as sample_tiles_main                           # for execution of the sample_tiles subcommand
    from .sample_tiles.config import _add_arguments as add_sample_tiles_arguments              # for the argument setup for the sample_tiles parser
    
    from .view_files.view_files import main as view_files_main                                 # for execution of the view_files subcommand
    from .view_files.config import _add_arguments as add_view_files_arguments                  # for the argument setup for the view_files parser
   
    

    # create the top-level parser
    parser = argparse.ArgumentParser(prog='patchcraft', description='A tool to sample and preprocess patches from slides preprocessed by pamly.')

    # Set up the subparsers, adding one for each subcommand
    subparsers = parser.add_subparsers(required=True)
    create_info_file_parser = subparsers.add_parser('create_info_file', help='Create file containing all necessary metadata for a directory comprised of slides in .sqlite format.')
    get_overview_parser = subparsers.add_parser('get_overview', help='Get an overview of the metadata contained in the info file.')
    sample_data_parser = subparsers.add_parser('sample_data', help='Sample patches from multiple slides and save them as single files. The patches can then be used for training a neural network.')
    sample_tiles_parser = subparsers.add_parser('sample_tiles', help='Sample tiles from multiple slides and save them as single files. The patches can then be used for training a neural network.')
    view_files_parser = subparsers.add_parser('view_files', help='Plot the first 100 patches of a directory to visualize the results of the preprocessing.')
  
    
    # Add arguments to each subparser
    add_create_info_file_arguments(create_info_file_parser)
    add_get_overview_arguments(get_overview_parser)
    add_sample_data_arguments(sample_data_parser)
    add_sample_tiles_arguments(sample_tiles_parser)
    add_view_files_arguments(view_files_parser)
   
    # Ensure each parser knows which function to call.
    # set_defaults can be used to set a new arg which isn't set on the command line.
    create_info_file_parser.set_defaults(main=info_main)
    get_overview_parser.set_defaults(main=get_overview_main)
    sample_data_parser.set_defaults(main=sample_data_main)
    sample_tiles_parser.set_defaults(main=sample_tiles_main)
    view_files_parser.set_defaults(main=view_files_main)

    # Extract command line arguments
    args = parser.parse_args()

    # run the appropriate function
    args.main(args)

# this ensure that both ways of calling the package work and don't interfer with each other
if __name__ == "__main__":
    main()

