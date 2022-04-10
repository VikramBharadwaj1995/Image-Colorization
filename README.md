## Vikram Bharadwaj - bharadwaj.vi@northeastern.edu


# Image Colorization Starter Code
The objective is to produce color images given grayscale input image. 

## Setup Instructions
Create a conda environment with pytorch, cuda. 

`$ conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia`

For systems without a dedicated gpu, you may use a CPU version of pytorch.
`$ conda install pytorch torchvision torchaudio cpuonly -c pytorch`

## Dataset
Use the zipfile provided as your dataset. You are expected to split your dataset to create a validation set for initial testing. Your final model can use the entire dataset for training. Note that this model will be evaluated on a test dataset not visible to you.

## Code Guide
Baseline Model: A baseline model is available in `basic_model.py` You may use this model to kickstart this assignment. We use 256 x 256 size images for this problem.
-	Fill in the dataloader, (colorize_data.py)
-	Fill in the loss function and optimizer. (train.py)
-	Complete the training loop, validation loop (train.py)
-	Determine model performance using appropriate metric. Describe your metric and why the metric works for this model? 
- Prepare an inference script that takes as input grayscale image, model path and produces a color image. 

## Additional Tasks 
- The network available in model.py is a very simple network. How would you improve the overall image quality for the above system? (Implement)
- You may also explore different loss functions here.

## Bonus
You are tasked to control the average color/mood of the image that you are colorizing. What are some ideas that come to your mind? (Bonus: Implement)

## Solution
- Document the things you tried, what worked and did not. 
- Update this README.md file to add instructions on how to run your code. (train, inference). 
- Once you are done, zip the code, upload your solution.  

# Documenting my progress for Image Colorization:
The following are the model parameters and training steps:</br>
- Got dataloaders to work by passing the train and validation dataset, by splitting the dataset into 80-20 train/test.
- Defined the following model parameters:
    * Learning rate = 0.001
    * Batch size = 64
    * Epochs = 100
    * Training/Val split = 80% train and 20% validation
    * Criteria = Mean Squared Error loss function
    * Optimizer = Tried both RMSProp and Adam and found Adam to be working better
    * Evaluation Metric - Peak Signal to Noise Ratio(PSNR)
- All the checkpoints are being stored in a directory called './checkpoints', in the current working directory. The train vs validation loss and PSNR plots will be stored in this directory once the training is complete.

The following are the methods I have used and have tried:</br>
1) The first method uses a simple CNN as given in the problem statement. The output of the model will be a color image, with 3 channels and I try to minimize the error by calculating the MSE between the traget or the acutal output image and the obtained output image from the model after running the inference step.</br>
2) The second method I would like to try in the future is using the L *a *b channels. According to the the following paper - https://arxiv.org/pdf/1603.08511.pdf the L channel can be used to predict the values of *a and *b channels. This is an interesting take and again, we can either use regression to predict the channel values or use classification to predict the bin(class) to which ab belong.

# Evaluation Metrics for the model:
1) Here, I have used Peak Signal to Noise Ratio(PSNR) as the performance evaluation metric for the model, which evaluates the difference between the original image and the obtained color image and is measured in decibels. A higher PSNR value indicates a higher reconstruction quality of the obtained image. Even though many colorization methods use CIELAB and YUV color spaces, the obtained results are transformed to RGB color space because RGB representation of images is a standard way to display colors. For this reason, PSNR between the R, G and B components of the original and the colorized image can be used as a performance metric for colorization.
2) The second evaluation metric I have used for improving image quality is SSIM(Structural Similarity Index) and is roughly based on - https://research.nvidia.com/sites/default/files/pubs/2017-03_Loss-Functions-for/NN_ImgProc.pdf. Here, I take in the input as the image obtained from the CNN and the actual target image and try to increase the structural similarity score and make it as close to 1 through backprop.


# Additional tasks:
3) I have also implemented a model that leverages a simple CNN to construct a higher resolution image from the given input image. This approach is based on the paper - https://arxiv.org/pdf/1501.00092v3.pdf. The model and the way to train this model is written in super_resolution.py. Although I have used a pretrained network to achieve this, the model size is pretty small and gives a better smoothened result.</br>
4) Here, as asked in the problem statement, I have tried to control the average temperature of the image using all 3 channels (R, G and B) using the GIMP algorithm. I have written the function gimp_color_balance which is roughly based on the following implementation - https://docs.gimp.org/2.10/en/gimp-layer-white-balance.html.</br>
5) The temperature or white-balance of the image can also be controlled by a neural network and is given in the following research paper - https://openaccess.thecvf.com/content_CVPR_2020/papers/Afifi_Deep_White-Balance_Editing_CVPR_2020_paper.pdf.
According to the paper, the model is made up of an encoder-decoder architecture and I would like to implement the model in the future.

# Running the model:
1) Running the train pipeline - </br>
`$ python execute_model <path_to_data_directory>`
2) Running the inference script - </br>
`$ python inference_script <path_to_checkpoint> <path_to_grayscale_image>`