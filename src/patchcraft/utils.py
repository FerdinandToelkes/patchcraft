import logging
import os


def start_logging_for_slide(output_config, general_transform_config, slide_dir, output_dir_for_patches_of_slide):
    """
    Start logging for the sampling process in single file mode. 
    """
    loglevel = output_config['log_level']
    # Set the logging level, convert to upper case to allow the user to specify --log=DEBUG or --log=debug
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    # Configure logging -> force=True fixed bug that the log file was not created
    log_name = os.path.join(output_dir_for_patches_of_slide,  'sampling.log')
    logging.basicConfig(filename=log_name, filemode='w', encoding='utf-8', level=numeric_level, format='%(levelname)s %(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', force=True) 
    # Log the output config (as info such that it is differentiable from the logging of detailed data specific to certain steps)
    logging.info('================================ Output Configuration: ================================')
    logging.info(f"Path to sampled data:                        {output_config['path']}")
    logging.info(f"Number of slides to be processed:            {output_config['number_of_slides']}")
    logging.info(f"Number of patches to be extracted per slide: {output_config['number_of_patches_per_slide']}")
    logging.info(f"Metadata saved for each slide:               {output_config['desired_metadata']}")
    logging.info(f"Log level was set to:                        {output_config['log_level']}")
    logging.info(f"Log filename:                                {log_name}")
    logging.info("-> default values see configs.py")
    # Log the transform configuration
    logging.info("============================ Transform Configuration: ==========================")
    logging.info(f"The stain to be sampled:                     {general_transform_config['sampling']['stain']}")
    logging.info(f"Patch size in micro meter:                   {general_transform_config['sampling']['patch_size_um']}")
    logging.info(f"Pixels per m of the WSI (default value):     {general_transform_config['sampling']['wsi_pixels_per_m']}")
    logging.info(f"Target size for tensors:                     {general_transform_config['sampling']['target_size']}")
    logging.info(f"Overlap of patches:                          {general_transform_config['sampling']['overlap']}")
    logging.info("-> default values see configs.py (TransformConfig class)")
    # Log the individual steps of data augmentation
    logging.info("============================== Data augmentation steps: ============================")
    logging.info("Random rotation, center crop, resize to target size, random horizontal flip, random vertical flip, random color jitter")
    logging.info("-> see function augment() in Preprocessing.py")
    # Log other parameters used for preprocessing
    logging.info("======================== Other parameters and informations: ========================")
    logging.info(f"Input directory (location of the slides):    {slide_dir}")

def start_logging_for_info_file(input_config, output_config):
    loglevel = output_config['log_level']
    # Set the logging level, convert to upper case to allow the user to specify --log=DEBUG or --log=debug
    numeric_level = getattr(logging, loglevel.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError('Invalid log level: %s' % loglevel)
    # Configure logging -> force=True fixed bug that the log file was not created
    name = output_config['path'] + '/' + input_config['info_filename'] + '.log'
    logging.basicConfig(filename=name, filemode='w', encoding='utf-8', level=numeric_level, format='%(levelname)s %(asctime)s: %(message)s', datefmt='%d/%m/%Y %H:%M:%S', force=True) 
    # Log the config for the creating of the info file
    logging.info('============================ Config for the creation of the info file ================================')
    logging.info(f"Path to the directory which contains all slides in .sqlite format: {input_config['path']}")
    logging.info(f"Name or prefix of the info file: {input_config['info_filename']}")
    logging.info("=============================== Other parameters and informations: ===================================")

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    """
    Call in a loop to create terminal progress bar.
    Args:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
    # Print New Line on Complete
    if iteration == total: 
        print()  



