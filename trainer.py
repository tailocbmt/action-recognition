import os
import time
import argparse

import numpy as np
import torch
from torch import nn, optim
from torchvision.models.video import r2plus1d_18
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import TripletFrameDataset
from network import R2Plus1DClassifier

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

def train_model(num_classes, csvPath, layer_sizes=[2, 2, 2, 2], num_epochs=45, save=True, path="model_data.pth.tar"):
    """Initalizes and the model for a fixed number of epochs, using dataloaders from the specified directory, 
    selected optimizer, scheduler, criterion, defualt otherwise. Features saving and restoration capabilities as well. 
    Adapted from the PyTorch tutorial found here: https://pytorch.org/tutorials/beginner/transfer_learning_tutorial.html

        Args:
            num_classes (int): Number of classes in the data
            directory (str): Directory where the data is to be loaded from
            layer_sizes (list, optional): Number of blocks in each layer. Defaults to [2, 2, 2, 2], equivalent to ResNet18.
            num_epochs (int, optional): Number of epochs to train for. Defaults to 45. 
            save (bool, optional): If true, the model will be saved to path. Defaults to True. 
            path (str, optional): The directory to load a model checkpoint from, and if save == True, save to. Defaults to "model_data.pth.tar".
    """


    # initalize the ResNet 18 version of this model
    # model = R2Plus1DClassifier(num_classes=num_classes, layer_sizes=layer_sizes).to(device)
    model = r2plus1d_18(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features,num_classes)
    model = model.to(device)

    criterion = nn.TripletMarginLoss(margin=1.0, p=2) # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # hyperparameters as given in paper sec 4.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    # prepare the dataloaders into a dict
    train_dataloader = DataLoader(TripletFrameDataset(csvPath), batch_size=4, shuffle=True, num_workers=4)
    # IF training on Kinetics-600 and require exactly a million samples each epoch, 
    # import VideoDataset1M and uncomment the following
    # train_dataloader = DataLoader(VideoDataset1M(directory), batch_size=32, num_workers=4)
    val_dataloader = DataLoader(TripletFrameDataset(csvPath, mode='val'), batch_size=4, num_workers=4)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader}

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0

    # check if there was a previously saved checkpoint
    if os.path.exists(path):
        # loads the checkpoint
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")

        # restores the model and optimizer state_dicts
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        
        # obtains the epoch the training is to resume from
        epoch_resume = checkpoint["epoch"]

    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
        # each epoch has a training and validation step, in that order
        for phase in ['train', 'val']:

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                scheduler.step()
                model.train()
            else:
                model.eval()


            for inputs in dataloaders[phase]:
                # move inputs and labels to the device the training is taking place on
                inputs = [input.to(device) for input in inputs]
                optimizer.zero_grad()

                # keep intermediate states iff backpropagation will be performed. If false, 
                # then all intermediate states will be thrown away during evaluation, to use
                # the least amount of memory possible.
                with torch.set_grad_enabled(phase=='train'):
                    A = model(inputs[0])
                    P = model(inputs[1])
                    N = model(inputs[2])
                    # we're interested in the indices on the max values, not the values themselves
                    loss = criterion(A, P, N)

                    # Backpropagate and optimize iff in training mode, else there's no intermediate
                    # values to backpropagate with and will throw an error.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   

                running_loss += loss.item() * 4

            epoch_loss = running_loss / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss}")

        # save the model if save=True
        if save:
            torch.save({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'loss': epoch_loss,
            'opt_dict': optimizer.state_dict(),
            }, path)

    # print the total time needed, HH:MM:SS format
    time_elapsed = time.time() - start    
    print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("numClasses", type=int)
    parser.add_argument("epoch", type=int,default=50)
    parser.add_argument("csvPath", type=str)
    parser.add_argument("savePath", type=str, default="model_data.pth.tar")
    args = parser.parse_args()
    train_model(num_classes=args.numClasses, csvPath=args.csvPath, num_epochs=args.epoch, path=args.savePath)
