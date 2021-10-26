import torch.utils.data as data
import pandas as pd
import numpy as np
import pickle
import torch
import os.path as osp
from mmaction.datasets.pipelines import Compose
import copy

class CustomDataModule(data.Dataset):
    def __init__(self, 
                cfg,
                csv_path, 
                kp_annotation, 
                sample_duration: int=48,
                mode: str='',
                **kwargs):
        super(CustomDataModule).__init__()

        self.cfg = cfg
        self.mode = mode
        self.sample_duration = sample_duration
        self.dataframe = pd.read_csv(csv_path) 
        self.dataframe = self.dataframe.replace(r'\\','/', regex=True)
        if self.mode not in  ['train', 'test', 'val']: 
            self.triplets = self.dataframe.loc[self.dataframe['status']==self.mode  , ['A', 'P', 'N']]
        else: 
            self.triplets = self.dataframe.loc[: , ['A', 'P', 'N']]

        with open(kp_annotation, 'rb') as f:
            self.kp_annotation = pickle.load(f)
        
        self.cache = {}
        # Number of sample clip
        self.num_clips = 1
        self.modality = 'RGB'
        self.start_index = 1
        self.transform = self.get_transform()

    def get_transform(self):
        
        self.pipelines = (self.cfg.data.train.pipeline if self.mode=='train' else self.cfg.data.test.pipeline)[1:-3]
        return Compose(self.pipelines)

    def _load_data(self, fname):
        frameDirPath = osp.join('/content',fname.split('.')[0])
        for i in range(len(self.kp_annotation)):
            if self.kp_annotation[i]['frame_dir'] == frameDirPath:
                buffer = copy.deepcopy(self.kp_annotation[i])
                buffer['num_clips'] = self.num_clips
                buffer['clip_len'] = self.sample_duration * (buffer['total_frames'] // self.sample_duration)
                buffer['modality'] = self.modality
                buffer['start_index'] = self.start_index
                return buffer   

    def _toTensor(self, buffer):
        buffer = np.transpose(buffer, (3, 0, 1, 2))
        buffer = np.expand_dims(buffer, axis=0)
        return torch.from_numpy(buffer)

    def _sample(self, buffer):
        ind = np.random.randint(buffer.shape[0]-self.sample_duration, size=1)[0]
        buffer = buffer[ind: ind+self.sample_duration, :, :,:]
        return buffer
       
    def __getitem__(self, index):
        tripletPath = self.triplets.iloc[index, :]
        triplets = []
        
        for path in tripletPath:
            buffer = None
            if path not in self.cache:
                buffer = self._load_data(path)
                buffer = self.transform(buffer)['imgs']
                self.cache[path] = buffer
            else: 
                buffer = self.cache[path]

            buffer = self._sample(buffer)
            buffer = self._toTensor(buffer)
            triplets.append(buffer)

        return triplets

    def __len__(self):
        return len(self.triplets)