import torch
import cv2
from SRCNN_model import SRCNN
from ssim_loss import calculate_ssim
import numpy as np
import os
from torchvision.utils import save_image
import sys

def exec_test(image_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = SRCNN().to(device)
    model.load_state_dict(torch.load("./srcnn_x3.pth"))
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    test_image_name = image_path.split(os.path.sep)[-1].split('.')[0]
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = image.reshape(image.shape[0], image.shape[1], 1)
    image = image / 255. # normalize the pixel values
    cv2.imshow('Greyscale image', image)
    cv2.waitKey(0)
    model.eval()
    with torch.no_grad():
        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        image = torch.tensor(image, dtype=torch.float).to(device)
        image = image.unsqueeze(0)
        image = image.reshape
        outputs = model(image)
    outputs = outputs[0].cpu()
    if not os.path.isdir('outputs'):
        os.mkdir('outputs')
    save_image(outputs, f"../outputs/SRCNN_{test_image_name}.jpg")

    print(f"The output of the model has been stored in the \"outputs\" directory as SRCNN_{test_image_name}.jpg")

if __name__ == '__main__':
    image_path = sys.argv[1]
    input_image = sys.argv[2]
    image_name = image_path.split(os.path.sep)[-1].split('.')[0]
    exec_test(image_path)
    print("SSIM Loss")
    SSIM_image, SSIM_score = calculate_ssim(image_path, input_image)
    print("Final SSIM value = ", SSIM_score)
    save_image(SSIM_image, f"../outputs/SSIM_{image_name}.jpg")