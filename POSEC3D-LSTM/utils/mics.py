import argparse
import torch
import copy
import pickle
import numpy as np
import os.path as osp

from torch.autograd import Variable
from sklearn.neighbors import NearestNeighbors
from mmaction.datasets.pipelines import Compose

def createKNN(arr, k):
    """
    Function to train an NearestNeighbors model, use to improve the speed of retrieving image from embedding database
    Args:
        X: data to train has shape MxN
        k: number of max nearest neighbors want to search
    
    Return:
        Nearest Neigbors model
    """
    model = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    model.fit(arr)
    return model

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfgPath", 
                        default='',
                        help="Config path")

    parser.add_argument("--csvPath", 
                        default='',
                        help="Csv path")

    parser.add_argument("--kpAnnotation", 
                        default='',
                        help="Annotation path contains keypoints")

    parser.add_argument("--txtPath", 
                        default='',
                        help="txt path to store video path(Used when run inference)")

    parser.add_argument("--logPath", 
                        default='action-recognition\src\R2Plus1D-PyTorch\POSEC3D-LSTM\log.csv',
                        help="Path to the csv log file")
    
    parser.add_argument("--embedPath", 
                        default='embeddings',
                        help="Path to the saved dataset embedding")

    parser.add_argument("--batch", 
                        default=8,
                        type=int,
                        help="Batch size (default: 8)")

    parser.add_argument("--device", 
                        default='cuda',
                        help="cuda or cpu")

    parser.add_argument("--workers", 
                        default=2,
                        type=int,
                        help="Number of worker (default: 8)")
    
    parser.add_argument("--save",
                        action='store_true', 
                        default=False, 
                        help="Save plot or not")

    parser.add_argument('--top', 
                    default=1,
                    type=int,
                    help='Top K nearest embedding')                    

    return parser.parse_args()

class PoseC3DTransform:
    def __init__(self,
                cfg,
                kp_annotation: str='',
                sample_duration: int=48,
                mode: str='test',
                num_clips: int=1,
                start_index: int=1,
                modality: str='RGB',
                seed: int=-1,
                ram: bool=True,
                ):
        self.cfg = cfg
        self.mode = mode
        self.sample_duration = sample_duration
        self.num_clips = num_clips
        self.start_index = start_index
        self.modality = modality
        self.seed = seed
        if ram:
            self.cache = {}

        self._getPipeline()
        with open(kp_annotation, 'rb') as f:
            self.kp_annotation = pickle.load(f)

    def _getPipeline(self):
        pipelines = (self.cfg.data.train.pipeline if self.mode=='train' else self.cfg.data.test.pipeline)[1:-3]
        self._preprocess = Compose(pipelines)
    
    def _toTensor(self, buffer):
        buffer = np.transpose(buffer, (3, 0, 1, 2))
        buffer = np.expand_dims(buffer, axis=0)
        return torch.from_numpy(buffer)

    def _sample(self, buffer):
        ind = np.random.randint(buffer.shape[0]-self.sample_duration, size=1)[0]
        buffer = buffer[ind: ind+self.sample_duration, :, :,:]
        return buffer

    def _load_video(self, path):
        frameDirPath = osp.join('/content',path.split('.')[0])
        for i in range(len(self.kp_annotation)):
            if self.kp_annotation[i]['frame_dir'] == frameDirPath:
                buffer = copy.deepcopy(self.kp_annotation[i])
                buffer['num_clips'] = self.num_clips
                buffer['clip_len'] = self.sample_duration * (buffer['total_frames'] // self.sample_duration)
                buffer['modality'] = self.modality
                buffer['start_index'] = self.start_index
                return buffer

    def _load_if_no_kp():
        pass

    def __call__(self, path):
        if not self.cache or path in self.cache:
            buffer = self.cache[path]
        else:
            buffer = self._load_video(path)
            buffer = self._preprocess(buffer)['imgs']
            buffer = self._sample(buffer)
            buffer = self._toTensor(buffer)
        
        return buffer

def print_model(model):
    params = model.to('cpu')
    for k,v in sorted(params.items()):
        print(k, v.shape)
        params[k] = Variable(torch.from_numpy(v), requires_grad=True)
    
    print('\nTotal parameters: ', sum(v.numel() for v in params.values()))