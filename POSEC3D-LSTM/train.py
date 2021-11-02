from mmcv import Config

from utils.mics import parse_args, PoseC3DTransform, createKNN
from utils.models import modelFromConfig
from utils.datasets import CustomDataModule

from torch.utils.data import DataLoader
from torch import nn, optim
import torch
import time
import os
import tqdm
import pandas as pd

def createDataloader(args):
    train_dataloader = DataLoader(CustomDataModule(cfg_path=args.cfg_path, 
                        csv_path=args.csv_path,
                        kp_annotation=args.kp_annotation),
                        batch_size=args.batch_size, 
                        shuffle=True, 
                        num_workers=args.workers)

    val_dataloader = DataLoader(CustomDataModule(cfg_path=args.cfg_path, 
                            csv_path=args.csv_path,
                            kp_annotation=args.kp_annotation,
                            mode='val'),
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.workers)

    test_dataloader = DataLoader(CustomDataModule(cfg_path=args.cfg_path, 
                            csv_path=args.csv_path,
                            kp_annotation=args.kp_annotation,
                            mode='test'),
                            batch_size=args.batch_size, 
                            shuffle=False, 
                            num_workers=args.workers)
    return train_dataloader, val_dataloader, test_dataloader

def createOptimizer(model, args):
    optimizer = optim.SGD(model.parameters(), lr=args.lr)  # hyperparameters as given in paper sec 4.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=args.gamma)  # the scheduler divides the lr by 10 every 10 epochs
    return optimizer, scheduler

def main():
    args = parse_args()
    
    config = Config.fromfile(args.cfg_path)
    config.merge_from_dict(args.cfg_options)

    # Load PoseC3D model
    model = modelFromConfig(cfg=config,checkpoint=args.checkpoint, device=args.device)

    # Create dataloader
    train_dataloader, val_dataloader, test_dataloader = createDataloader(args)
    dataloaders = {'train': train_dataloader, 'val': val_dataloader, 'test': test_dataloader}
    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val', 'test']}

    #create loss and optimizer
    criterion = nn.TripletMarginLoss(margin=1.0, p=2) 
    opt, scheduler = createOptimizer(model, args)

    # Training phase
    start = time.time()
    epoch_resume = 0
    log_file = []

    if os.path.exists(args.checkpoint):
        # loads the checkpoint
        checkpoint = torch.load(args.checkpoint)
        print("Reloading from previously saved checkpoint")
        # restores the model and optimizer state_dicts
        opt.load_state_dict(checkpoint['opt_dict'])
        # obtains the epoch the training is to resume from
        epoch_resume = checkpoint["epoch"]

    for epoch in range(epoch_resume, args.epochs):
        loss_log = []
        # each epoch has a training and validation step, in that order
        with tqdm(total=args.epochs, unit=" epoch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}")
            
            for phase in ['train', 'val','test']:
                # reset the running loss and corrects
                running_loss = 0.0

                # set model to train() or eval() mode depending on whether it is trained
                # or being validated. Primarily affects layers such as BatchNorm or Dropout.
                if phase == 'train':
                    scheduler.step()
                    model.train()
                else:
                    model.eval()


                for inputs in dataloaders[phase]:
                    # move inputs and labels to the device the training is taking place on
                    inputs = [input.to(args.device) for input in inputs]
                    opt.zero_grad()

                    # keep intermediate states iff backpropagation will be performed. If false, 
                    # then all intermediate states will be thrown away during evaluation, to use
                    # the least amount of memory possible.
                    with torch.set_grad_enabled(phase=='train'):
                        A = model(inputs[0], -1)
                        P = model(inputs[1], -1)
                        N = model(inputs[2], -1)
                        # we're interested in the indices on the max values, not the values themselves
                        loss = criterion(A, P, N)

                        # Backpropagate and optimize iff in training mode, else there's no intermediate
                        # values to backpropagate with and will throw an error.
                        if phase == 'train':
                            loss.backward()
                            opt.step()   

                    running_loss += loss.item()

                epoch_loss = running_loss / dataset_sizes[phase]

                loss_log.append(epoch_loss)
                print(f"{phase} Loss: {epoch_loss}")

            tepoch.update(1)
            tepoch.set_postfix(train_loss=loss.loss_log[0], val_loss=loss.loss_log[1], test_loss=loss.loss_log[2])
            # Append loss to log after each epoch
            log_file.append(loss_log)
            # save the model if save=True
            if args.save:
                torch.save({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'loss': epoch_loss,
                'opt_dict': opt.state_dict(),
                }, '{}/epoch_{}.pth.tar'.format(args.checkpoint_dir, epoch+1))

    # print the total time needed, HH:MM:SS format

    pd.DataFrame(log_file, columns=['train', 'val', 'test']).to_csv(args.log, index=False)

    time_elapsed = time.time() - start
    print('\n')
    print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")
if __name__ == "__main__":
    main()