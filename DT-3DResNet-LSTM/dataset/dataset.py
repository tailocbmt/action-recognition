import os
import cv2
import pickle
from numpy.core.defchararray import index
import pandas as pd
import torch
import torch.utils.data as data
from torchvision.transforms import (
    Compose,
    Lambda,
    RandomCrop,
    RandomHorizontalFlip,
    CenterCrop,
)
from pytorchvideo.transforms import (
    Normalize,
    RandomShortSideScale,
    ShortSideScale,
)

from .temporal_transforms import TemporalSequenceCrop
from .pose_transforms import PoseDecode, GeneratePoseTarget

class CustomDataModule(data.Dataset):
    def __init__(self, 
                csv_path: str='', 
                img_path: str='',
                kp_annotation: str='', 
                sample_duration: int=16,
                mode: str='train',
                **kwargs):
        super(CustomDataModule).__init__()
        self.img_path = img_path
        with open(kp_annotation, 'rb') as f:
            self.kp_annotation = pickle.load(f)

        self.dataframe = pd.read_csv(csv_path) 
        self.dataframe = self.dataframe.replace(r'\\','/', regex=True)
        self.triplets = self.dataframe.loc[self.dataframe['status']==mode, ['A', 'P', 'N']]
        self.cache = {}
        
        self.mode = mode
        self.min_size = 256
        self.max_size = 320
        self.mean = (0.45, 0.45, 0.45)
        self.std = (0.225, 0.225, 0.225)
        self.crop_size = 224

        self.temporal_transform = TemporalSequenceCrop(sample_duration)
        self.pose_decode = PoseDecode()
        self.generate_pose = GeneratePoseTarget(
            sigma=0.6,
            use_score=True,
            with_kp=True,
            with_limb=False
        )
        self.transform = self.get_transform()

    def get_transform(self):
        
        return Compose([
                Normalize(self.mean, self.std),
                Lambda(lambda x: x / 255.0),
            ]
            + (
                [
                    RandomShortSideScale(
                        min_size=self.min_size,
                        max_size=self.max_size,
                    ),
                    RandomCrop(self.crop_size),
                    RandomHorizontalFlip(p=0.5),
                ]
                if self.mode == 'train'
                else [
                    ShortSideScale(self.min_size),
                    CenterCrop(self.crop_size),
                ])
        )

    def load_kpheatmap(self, fname):
        frameDirPath = os.path.join('/content',fname.split('.')[0])
        
        for i in range(len(self.kp_annotation)):
            if self.kp_annotation[i]['frame_dir'] == frameDirPath:
                buffer = self.pose_decode(self.kp_annotation[i])

                return buffer   

        
    def __getitem__(self, index):
        tripletPath = self.triplets.iloc[index, :]
        triplets = []
        
        for path in tripletPath:
            buffer = None
            if path not in self.cache:
                buffer = self.load_kpheatmap(path)
                self.cache[path] = buffer
            else: 
                buffer = self.cache[path]
                buffer = self.temporal_transform(buffer, index)
                buffer = self.crop(buffer, index)
                buffer = self.transform(buffer)
                
            triplets.append(buffer)

        return triplets