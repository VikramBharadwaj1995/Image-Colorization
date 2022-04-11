import os
from typing import Tuple

# Import PIl.Image to read image data
import cv2
import torch
import torchvision.transforms as T
from torch.utils.data import Dataset
from torchvision import transforms


class ColorizeData(Dataset):
    def __init__(self, landscape_dataset, data_directory):
        # Initialize dataset, you may use a second dataset for validation if required
        # Use the input transform to convert images to grayscale
        self.landscape_dataset = landscape_dataset
        self.data_directory = data_directory
        self.input_transform = T.Compose([T.ToTensor(),
                                          T.Resize(size=(224, 224)),
                                          T.Normalize((0.5), (0.5))
                                          ])
        # Use this on target images(colorful ones)
        self.target_transform = T.Compose([T.ToTensor(),
                                           T.Resize(size=(224, 224)),
                                           T.Normalize((0.5, 0.5), (0.5, 0.5))])

    def __len__(self) -> int:
        # return Length of dataset
        return len(self.landscape_dataset)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Return the input tensor and output tensor for training
        image_path = os.path.join(
            self.data_directory, self.landscape_dataset[index])

        color_image = cv2.imread(image_path)

        normalize = transforms.ToTensor()
        l_channel, a, b = cv2.split(color_image)
        l_channel = normalize(self.input_transform(l_channel))
        ab = normalize(cv2.merge((a, b)))
        ab = self.target_transform(ab)
        return (l_channel, ab)
