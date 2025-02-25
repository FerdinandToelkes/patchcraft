import torch
import torchvision.transforms.v2 as T 
import torchvision.transforms.v2.functional as F
import numpy as np
import logging
import cv2
import math





def augment_patch(one_patch_np: np.ndarray, general_transform_config: dict, training_transform_config: dict) -> torch.Tensor:
    """Transform numpy array to torch tensor and augment the image by applying random rotation, center crop, resize, random horizontal and vertical flip, and random color jitter.
    
    Args:
        one_patch_np (np.ndarray): Numpy array of the patch.
        general_transform_config (dict): General transformation configuration.
        training_transform_config (dict): Training transformation configuration.
    
    Returns:
        torch.Tensor: Augmented image as torch tensor.
    """
    # convert to tensor
    tensor = to_torch_tensor(one_patch_np)
    # apply augmentation
    patch_size_in_pixels_pre_rotation = tensor.size()[1]
    if training_transform_config['rotation']['enabled'] is True:
        tensor = random_rotation(tensor)
    tensor = center_crop(tensor, patch_size_in_pixels_pre_rotation)
    ts = general_transform_config['sampling']['target_size']
    tensor = resize(tensor, target_size=[ts,ts])
    if training_transform_config['flips']['enabled'] is True:
        tensor = random_horizontal_flip(tensor)
        tensor = random_vertical_flip(tensor)
    tensor = random_color_jitter(tensor, training_transform_config)
    return tensor

def augment_tile(one_patch_np: np.ndarray, training_transform_config: dict) -> torch.Tensor:
    """ Transform numpy array to torch tensor and augment the image by applying random horizontal and vertical flip, and random color jitter.
    
    Args:
        one_patch_np (np.ndarray): Numpy array of the patch.
        training_transform_config (dict): Training transformation configuration.

    Returns:
        torch.Tensor: Augmented image as torch tensor.
    """
    # convert to tensor
    tensor = to_torch_tensor(one_patch_np)
    # apply augmentation
    if training_transform_config['flips']['enabled'] is True:
        tensor = random_horizontal_flip(tensor)
        tensor = random_vertical_flip(tensor)
    tensor = random_color_jitter(tensor, training_transform_config)
    return tensor

#################################### Utility functions for augment() ##########################################
def to_torch_tensor(one_patch_np: np.ndarray) -> torch.Tensor:
    """ Convert numpy array to torch tensor.

    Args:
        one_patch_np (np.ndarray): Numpy array of the patch.

    Returns:
        torch.Tensor: Image as torch tensor.
    """
    if one_patch_np is None:
        logging.error("Empty cannot be converted to tensor")
        raise Exception("Empty cannot be converted to tensor")
    # Note on the dimensions: Numpy: (H, W, C) -> slicing logic in get_tiles_and_combine_to_patch(): (W, H, C) -> pytorch tensor (C, H, W)
    return torch.from_numpy(np.transpose(one_patch_np, (2,1,0))) # pytorch wants (batch, color, height, width)

def random_rotation(tensor: torch.Tensor) -> torch.Tensor:
    """ Random rotation of the image by 0-360 degrees.
    
    Args:
        tensor (torch.Tensor): Image as torch tensor.

    Returns:
        torch.Tensor: Rotated image as torch tensor.
    """
    rotater = T.RandomRotation(degrees=(0, 360))
    return rotater(tensor)

def center_crop(tensor: torch.Tensor, patch_size_in_pixels_pre_rotation: int) -> torch.Tensor:
    """ Center crop the image to the desired patch size.
    
    Args:
        tensor (torch.Tensor): Image as torch tensor.
        patch_size_in_pixels_pre_rotation (int): Patch size in pixels before rotation.

    Returns:
        torch.Tensor: Center cropped image as torch tensor.
    """
    patch_size_in_pixels = math.floor(patch_size_in_pixels_pre_rotation / math.sqrt(2)) # get real patch size, factor sqrt(2) was introduced in calculate_patch_size()
    output_size = [patch_size_in_pixels, patch_size_in_pixels]
    return F.center_crop(tensor, output_size) # input shape has to be (channels, height, width)

def resize(tensor: torch.tensor, target_size: list[int]) -> torch.Tensor:
    """ Resize the image to the desired target size, i.e. the input size (H,W) for the inteded neural network.
    
    Args:
        tensor (torch.Tensor): Image as torch tensor.
        target_size (list[int]): Target size of the image.

    Returns:
        torch.Tensor: Resized image as torch tensor.
    """
    one_patch_np = np.transpose(tensor.numpy(), (1, 2, 0)) # since we changed the dimension in to_torch_tensor
    resized_array = cv2.resize(one_patch_np, dsize=target_size, interpolation=cv2.INTER_LANCZOS4)
    return to_torch_tensor(resized_array.clip(max=255).astype('uint8')) # make sure that output is valid rgb tensor

def random_horizontal_flip(tensor: torch.Tensor, p=0.5) -> torch.Tensor:
    """ Randomly flip the image horizontally. 
    
    Args:
        tensor (torch.Tensor): Image as torch tensor.
        p (float): Probability of the flip.

    Returns:
        torch.Tensor: Horizontally flipped image as torch tensor.
    """
    transform = T.RandomHorizontalFlip(p=p)
    return transform(tensor)

def random_vertical_flip(tensor: torch.Tensor, p=0.5) -> torch.Tensor:
    """ Randomly flip the image vertically. 
    
    Args:
        tensor (torch.Tensor): Image as torch tensor.
        p (float): Probability of the flip.

    Returns:
        torch.Tensor: Vertically flipped image as torch tensor.
    """
    transform = T.RandomVerticalFlip(p=p)
    return transform(tensor)
    
def random_color_jitter(tensor: torch.Tensor, training_transform_config: dict) -> torch.Tensor:
    """ Randomly change the brightness, contrast, saturation and hue of the image. 
    
    Args:
        tensor (torch.Tensor): Image as torch tensor.
        training_transform_config (dict): Training transformation configuration.
        
    Returns:
        torch.Tensor: Color jittered image as torch tensor.
    """
    jitter = T.ColorJitter(brightness = training_transform_config['color_jitter']['brightness_jitter'], 
                            contrast = training_transform_config['color_jitter']['contrast_jitter'],
                            saturation = training_transform_config['color_jitter']['saturation_jitter'],
                            hue = training_transform_config['color_jitter']['hue_jitter'])
    return jitter(tensor)
