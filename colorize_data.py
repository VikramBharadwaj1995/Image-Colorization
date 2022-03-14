from email.mime import image
from logging import root
from typing import Tuple
from torch.utils.data import Subset
import torch
import torchvision
import os
from PIL import Image
import numpy as np
from skimage.color import rgb2lab, rgb2gray

class ColorizeData(Subset):
    def __init__(self, root_dir = 'landscape_images', subset = Subset, input_transform = torchvision.transforms, target_transform = torchvision.transforms):
        # Initialize dataset, you may use a second dataset for validation if required
        self.root_directory = root_dir
        # Get size of the dataset, which is the total number of images = 4282
        self.dataset_size = len(os.listdir(root_dir))
        # Transform the input images
        self.input_transform = input_transform
        self.target_transform = target_transform

    def __len__(self) -> int:
        # return Length of dataset
        return self.dataset_size
    
    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        dataset = os.listdir(self.root_directory)
        image_name = os.path.join(self.root_directory, dataset[index])
        image_original = Image.open(image_name).convert("RGB")

        # Convert the image from RGB space to L*a*b color space
        img_lab = rgb2lab(np.asarray(image_original))
        # Normalize the image
        img_lab = (img_lab + 128) / 255
        # Extract *a*b channels
        img_ab = img_lab[:, :, 1:3]
        # target_transform on *a*b channels
        img_ab = self.target_transform(img_ab).float()
        # input_transform on the L channel
        image_l = self.input_transform(image_original).float()

        return image_l, img_ab