# Name - Vikram Bharadwaj<br>Email - vikrambharadwaj1995@gmail.com

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

## Documenting my progress
***Please find comments in every file about the use of every line of code written!***
The approach I am following is as follows:
Any image contains the following channels: L, *a and *b.
L - This channel always corresponds to the greyscale component of the image or the lightness of the image.
*a and *b channels - Map the four unique colors Red, Green, Blue and Yellow.

- Made changes to basic_model.py by adding only 2 output neurons from the final layer of the convolutional network, which maps to *a and *b channels. 
- Made changes to basic_model.py by adding an extra class called AverageMeter(), which tracks the metrics of the model such as training loss, validation loss and and time taken for each batch.
- Added the use_gpu flag to check for GPU, else use the model as is on the available CPU
- Got dataloaders to work by passing the train and validation dataset
- Defined the following model parameters:
    * Learning rate = 0.001
    * Batch size = 64
    * Epochs = 50
    * Training/Val split = 75% train and 25% validation
    * Criteria = Mean Squared Error loss function
    * Optimizer = Tried both RMSProp and Adam and found Adam to be working better
- The training and validation methods convert the incoming input and output into a pytorch Variable, that helps in conversion of the incoming data to a pytorch wrapper around the tensor.
- To enhance the final input image, I tried playing around with different paramters and functions that OpenCV has to offer. I ended up using cv2.detailEnhance, which worked much better than adding any kind of filter, dilation or erosion of the image.

## Running the model
- Please run the model to train as:
` python execute_model.py <data_dir> or by default landscape_images will be picked `
- Please run the inference script as:
` python inference.py <checkpoint_file> <grayscale_image>` -> Stores a file called output.jpg, which will be the colored image.

## Bonus
- Since I have used the L *a and *b channels to work on this challenge, I believe the rescaling done at the end for the lighting channel will handle the general color temperature/lighting factor of the image.