import pytorch_ssim
import torch
from torch.autograd import Variable
from torch import optim
import cv2
import numpy as np

def calculate_ssim(output_image, target_image):
    img1 = torch.from_numpy(np.rollaxis(target_image, 2)).float().unsqueeze(0)/255.0
    img2 = torch.rand(output_image.size())

    if torch.cuda.is_available():
        img1 = img1.cuda()
        img2 = img2.cuda()


    img1 = Variable( img1, requires_grad=False)
    img2 = Variable( img2, requires_grad=True)


    ssim_value = pytorch_ssim.ssim(img1, img2).data[0]
    print("Initial SSIM:", ssim_value)

    # Define the loss to be SSIM() loss
    ssim_loss = pytorch_ssim.SSIM()

    # Define Adam optimizer
    optimizer = optim.Adam([img2], lr=0.01)

    # While SSIM value is less than 95% similarity, backpropogagte and reduce loss
    while ssim_value < 0.95:
        optimizer.zero_grad()
        ssim_out = -ssim_loss(img1, img2)
        ssim_value = - ssim_out.data[0]
        print("SSIM score = ", ssim_value)
        ssim_out.backward()
        optimizer.step()

    final_image = np.transpose(img2.cpu().detach().numpy()[0],(1,2,0))
    return final_image, ssim_value
