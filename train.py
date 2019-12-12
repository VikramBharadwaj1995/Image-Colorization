from pyexpat import model
import torch
from colorize_data import ColorizeData
from basic_model import Net, AverageMeter
import torch.nn as nn
import time, os
from torch.autograd import Variable
import torchvision.transforms as T

class Trainer:
    def __init__(self):
        self.workers = 4
        self.epochs = 50
        self.batch_size = 64
        self.learning_rate = 0.001
        self.train_split = 0.75

    def init_model_and_data(self, data_dir, use_gpu):
        # Inittialize dataset - ColorizeData()
        dataset = os.listdir(data_dir)

        # Split into train and test data
        train_size = int(self.train_split * len(dataset))
        test_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

        # Transforms
        train_transform = T.Compose([T.ToTensor(),
                                        T.Resize(size=(256,256)),
                                        T.Grayscale(),
                                        T.RandomHorizontalFlip()])

        target_transform = T.Compose([T.ToTensor(),
                                        T.Resize(size=(256,256))])

        # Datasets
        train_dataset = ColorizeData(data_dir, train_dataset, train_transform, target_transform)
        val_dataset = ColorizeData(data_dir, val_dataset, train_transform, target_transform)

        # Dataloader - Pytorch Dataloaders are being used to instantiate the training and validation dataset
        train_dataloader = torch.utils.data.DataLoader(train_dataset, self.batch_size, shuffle=True)
        val_dataloader = torch.utils.data.DataLoader(val_dataset, self.batch_size, shuffle=False)

        # Model
        model = Net()
        # Loss function to use
        if use_gpu:
            model.cuda()
            criterion = nn.MSELoss().cuda()
            print(model)
            print("Model loaded on the GPU -> ", torch.cuda.get_device_name(0))
        else:
            print(model)
            criterion = nn.MSELoss()
        
        # You may also use a combination of more than one loss function or create your own.
        optimizer = torch.optim.Adam(model.parameters(), lr=self.learning_rate)

        return model, train_dataloader, val_dataloader, criterion, optimizer, self.epochs
        
    def save_checkpoint(self, state, flag, filename='checkpoints/checkpoint.pth.tar'):
        torch.save(state, filename)
    
    def train_model(self, model, train_dataloader, criterion, optimizer, current_epoch, use_gpu):
        # Initialize value counters and performance metrics for the model
        batch_time = AverageMeter()
        losses = AverageMeter()

        # Train loop
        model.train()

        # Time taken for training
        end = time.time()

        # Calculate loss
        for i, (input_l, target) in enumerate(train_dataloader):
            # Use GPU if available
            if use_gpu:
                input_l = Variable(input_l).cuda()
                target = Variable(target).cuda()
            else:
                input_l = Variable(input_l)
                target = Variable(target)

            output_ab = model(input_l)
            # Calculate the loss.
            loss = criterion(output_ab, target)
            
            losses.update(loss.item(), input_l.size(0))
            
            # Set the gradient of tensors to zero.
            optimizer.zero_grad()
            # Computes the gradient of the current tensor by employing chain rule and propogating it back in the network.
            loss.backward()
            # Update the parameters in the direction of the gradient.
            optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            if i % 25 == 0:
                print("Current loss = ", loss)
                print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        current_epoch, i, len(train_dataloader), batch_time=batch_time,
                        loss=losses))


    def validate(self, model, val_dataloader, criterion, current_epoch, use_gpu):
        # Validation loop begin
        # ------
        # Initialize value counters and performance metrics for the model
        batch_time = AverageMeter()
        losses = AverageMeter()

        # Train loop
        model.eval()

        # Time taken for evaluation
        end = time.time()

        # Calculate loss
        for i, (input_l, target) in enumerate(val_dataloader):
            
            # Use GPU if available
            if use_gpu:
                input_l = Variable(input_l).cuda()
                target = Variable(target).cuda()
            else:
                input_l = Variable(input_l)
                target = Variable(target)

            output_ab = model(input_l)
            loss = criterion(output_ab, target)
            
            losses.update(loss.item(), input_l.size(0))

            batch_time.update(time.time() - end)
            end = time.time()

            # Print epoch information
            if i % 25 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                        'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                        'Loss {loss.val:.4f} ({loss.avg:.4f})\t'.format(
                        current_epoch, i, len(val_dataloader), batch_time=batch_time,
                        loss=losses))
        
        # Validation loop end
        # ------
        print("Validation completed!")
        return losses.avg