import os
import argparse
import torch
import pickle
from torchvision.models.video import r2plus1d_18
from torch import nn
from torch.utils.data import DataLoader

from dataset import VideoDataset

def interfere(ckpt_pth):
  #DatabasePath
  database_dir = os.path.join(os.getcwd(),'action')
  vid_dir = os.path.join(os.getcwd(),'action')

  device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
  print("Device being used:", device)
  
  #R2plus1D model
  num_classes = 1000
  model = r2plus1d_18(pretrained=True)
  num_features = model.fc.in_features
  model.fc = nn.Linear(num_features,num_classes)
  model = model.to(device)

  if os.path.exists(ckpt_pth):
      checkpoint = torch.load(ckpt_pth,map_location=torch.device('cpu'))
      model.load_state_dict(checkpoint['state_dict'])
      print("Successfully loaded the desired checkpoint")
  else: print("Checkpoint path invalid")

  #Dataloader
  train_dataloader = DataLoader(VideoDataset(database_dir,mode='train',clip_len=16),batch_size=1,num_workers=4)
  test_dataloader = DataLoader(VideoDataset(vid_dir,mode='test',clip_len=16),batch_size=1,num_workers=4)

  # Get database vector and test vector 
  base_outputs,base_labels,outputs,labels = [],[],[],[]
  with torch.no_grad():
    for input, label in train_dataloader:
      input = input.to(device)
      label = label.to(device)
      output = model(input)
      base_outputs.append(output)
      base_labels.append(label)

    for input,label in test_dataloader:
      input = input.to(device)
      label = label.to(device)
      output = model(input)
      outputs.append(output)
      labels.append(label)

  # Calculate Euclidean distance between test video and database vector
  dist_list = []
  for i in range(len(outputs)):
      dist=[]
      for j in range(len(base_outputs)):
        result = torch.cdist(outputs[i], base_outputs[j])
        dist.append(result)
      dist_list.append(dist)

  # Calculate topk accuracy
  def topk_accuracy(topk):
    count = 0
    for i in range(len(outputs)):
      for j in range(topk):
        index = dist_list[i].index(sorted(dist_list[i])[j])
        if labels[i] == base_labels[index]: 
          count+=1
          break
    prob = (count/len(outputs))*100
    return prob
    
  #Calculate top1, top2, top5 accuracy
  top1_prob = topk_accuracy(1)
  top2_prob = topk_accuracy(2)
  top5_prob = topk_accuracy(5) 
  print('Top 1 accuracy: {}%'.format(round(top1_prob,2)))
  print('Top 2 accuracy: {}%'.format(round(top2_prob,2)))
  print('Top 5 accuracy: {}%'.format(round(top5_prob,2)))
  

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("checkpoint_pth", type= str)
  # parser.add_argument("clip_len", type=int, default=16)
  args = parser.parse_args()
  interfere(ckpt_pth=args.checkpoint_pth)
