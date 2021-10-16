import os
import time
import argparse

import numpy as np
import pandas as pd
import torch
from torch import nn, optim
from torchvision.models.video import r2plus1d_18
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset.dataset import CustomDataModule
from models.models import CNN3DLSTM

# Use GPU if available else revert to CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device being used:", device)

def train_model(csvPath,num_classes=512, mode='train', num_epochs=45, save=True,batch_size=1, sample_duration=8, path="checkpoints/epoch_23.pth.tar",saveLog='log.csv'):
    
    model = CNN3DLSTM(pretrained=True,
                    hidden_dim=num_classes,
                    num_clip=sample_duration)

    criterion = nn.TripletMarginLoss(margin=1.0, p=2) # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=0.001)  # hyperparameters as given in paper sec 4.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs
    
    log_file = []
    if mode == 'train':
        train_dataloader = DataLoader(CustomDataModule(csv_path=csvPath, sample_duration=sample_duration),
                                    batch_size=batch_size, 
                                    shuffle=True, 
                                    num_workers=4)

        val_dataloader = DataLoader(CustomDataModule(csv_path=csvPath, sample_duration=sample_duration, mode='val'), 
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    num_workers=4)
        
        test_dataloader = DataLoader(CustomDataModule(csv_path=csvPath, sample_duration=sample_duration, mode='val'), 
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    num_workers=4)

        dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}

        dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}

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
            loss_log = []
            # each epoch has a training and validation step, in that order
            for phase in ['train', 'val','test']:

                # reset the running loss and corrects
                running_loss = 0.0

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

                    running_loss += loss.item() * batch_size

                epoch_loss = running_loss / dataset_sizes[phase]

                loss_log.append(epoch_loss)
                print(f"{phase} Loss: {epoch_loss}")

            # Append loss to log after each epoch
            log_file.append(loss_log)
            # save the model if save=True
            if save:
                torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': epoch_loss,
                'opt_dict': optimizer.state_dict(),
                }, 'checkpoint/epoch_{}.pth.tar'.format(epoch+1))

        # print the total time needed, HH:MM:SS format

        pd.DataFrame(log_file, columns=['train', 'val', 'test']).to_csv(saveLog, index=False)
        
        time_elapsed = time.time() - start
        print('\n')
        print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")
    
    elif mode == 'test':
        print('Testing mode....')
        test_dataloader = DataLoader(CustomDataModule(csv_path=csvPath, sample_duration=sample_duration, mode='val'), 
                                    batch_size=batch_size, 
                                    shuffle=False,
                                    num_workers=4)

        running_loss = 0.0
        model.eval()
        for inputs in test_dataloader:
            inputs = [input.to(device) for input in inputs]
            with torch.no_grad():
                A = model(inputs[0])
                P = model(inputs[1])
                N = model(inputs[2])

                loss = criterion(A, P, N)

            running_loss += loss.item() * batch_size

        epoch_loss = running_loss / len(test_dataloader)
        print(f"Test Loss: {epoch_loss}")

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("numClasses", type=int)
    parser.add_argument("epoch", type=int,default=50)
    parser.add_argument("csvPath", type=str)
    parser.add_argument("savePath", type=str, default="model_data.pth.tar")
    parser.add_argument("mode", type=str, default='train')
    args = parser.parse_args()
    train_model(num_classes=args.numClasses, csvPath=args.csvPath, num_epochs=args.epoch, path=args.savePath, mode=args.mode)