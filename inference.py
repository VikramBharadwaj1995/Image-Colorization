from email.mime import image
from tabnanny import check
import PIL
import torch
from matplotlib import pyplot as plt
from torchvision import transforms
from skimage.color import rgb2gray, lab2rgb
from basic_model import Net
import sys, cv2

if __name__ == '__main__':
    # Read the two input parameters, which is the model checkpoint and grayscale image
    model_checkpoint, gray_image = sys.argv[1], sys.argv[2]
    # Load model from basic_model.py by calling the class constructor
    model = Net()
    # If GPU available, set current execution to the CUDA instance, else, use CPU 
    if torch.cuda.is_available():
        model.cuda()
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # Load checkpoint
    checkpoint = torch.load(model_checkpoint)
    model.load_state_dict(checkpoint['state_dict'], strict=False)
    
    # Open the grayscale image
    img = PIL.Image.open(gray_image)
    # Reshape it to the same size as the one that is accepted by the network
    image_l = img.resize((256, 256))
    # Although a grayscale image, it might have 3 channels, so check for number of channels. If three, convert to grayscale
    if len(image_l.getbands()) == 3:
        image_l = rgb2gray(image_l)
    # Use the transform to convert to a float tensor 
    image_l = transforms.ToTensor()(image_l).float()
    
    # Evaluate the image from the model
    model.eval()
    
    with torch.no_grad():
        # Get the predicted values of *a and *b channels
        preds = model(image_l.unsqueeze(0).to(device))

    # Get the first element from the tensor, move from GPU to CPU and convert to float
    preds = preds[0].cpu().float()
    image_l = image_l.cpu()

    # Combine channels L and *a *b
    color_image = torch.cat((image_l, preds), 0).numpy()
    color_image = color_image.transpose((1, 2, 0))
    # Rescale
    color_image[:, :, 0:1] = color_image[:, :, 0:1] * 100
    color_image[:, :, 1:3] = color_image[:, :, 1:3] * 255 - 128
    # Convert to RGB
    color_image = lab2rgb(color_image)

    # Improve the final quality of the image using detailEnhance from cv2 package.
    final_image = cv2.detailEnhance(color_image, sigma_s = 10, sigma_r = 0.15)
    plt.imsave('output.jpg', final_image)