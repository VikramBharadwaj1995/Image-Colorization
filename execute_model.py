from operator import imod
from train import Trainer
import torch
import sys

def main(data_dir):
    
    # Initialize training object from the Train class
    trainer_object = Trainer()
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    
    # Use GPU if CUDA available
    use_gpu = torch.cuda.is_available()
    
    # Initilize model
    model, train_dataloader, val_dataloader, criterion, optimizer, epochs = trainer_object.init_model_and_data(data_dir, use_gpu)
    
    # Initialize best loss
    current_best_loss = 1000.0
    
    # Begin training
    for epoch in range(0, epochs):
        #Train
        trainer_object.train_model(model, train_dataloader, criterion, optimizer, epoch, use_gpu)
        print("Training completed, epoch - ", epoch)
        
        # Calculate loss
        current_loss = trainer_object.validate(model, val_dataloader, criterion, epoch, use_gpu)
        if current_loss <= current_best_loss:
            current_best_loss = current_loss
        current_best_loss = max(current_best_loss, current_loss)
        flag = current_loss < current_best_loss
        
        # Save model checkpoint
        trainer_object.save_checkpoint({
            'epoch': epoch + 1,
            'current_best_loss': current_best_loss,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, flag, 'checkpoints/checkpoint-epoch-{}.pth.tar'.format(epoch))

    print("Training and validation complete - current loss = ", current_best_loss)
    return current_best_loss
    
if __name__ == '__main__':
    data_dir = sys.argv[1]
    print("Starting the execution!")
    main(data_dir)