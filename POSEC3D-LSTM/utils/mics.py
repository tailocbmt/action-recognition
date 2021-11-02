import argparse
import torch
import copy
import pickle
import numpy as np
import os.path as osp

from mmcv import DictAction
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
    parser.add_argument("--cfg-path", 
                        default='',
                        help="posec3d config file path")

    parser.add_argument("--csv-path", 
                        default='',
                        help="Csv path")

    parser.add_argument("--kp-annotation", 
                        default='',
                        help="Annotation path contains keypoints")

    parser.add_argument("--txt-path", 
                        default='',
                        help="txt path to store video path(Used when run inference)")

    parser.add_argument("--log-path", 
                        default='action-recognition\src\R2Plus1D-PyTorch\POSEC3D-LSTM\log.csv',
                        help="Path to the csv log file")
    
    parser.add_argument('--checkpoint', 
                    default='',
                    help='Path to the checkpoint')    

    parser.add_argument("--embed-path", 
                        default='embeddings',
                        help="Path to the saved dataset embedding")


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

    ####################################
    # Long demo
    parser.add_argument('--video', help='video file/url', default='')
    
    parser.add_argument('--out-filename', help='output filename', default='')

    parser.add_argument(
        '--det-config',
        default='demo/faster_rcnn_r50_fpn_2x_coco.py',
        help='human detection config file path (from mmdet)')

    parser.add_argument(
        '--det-checkpoint',
        default=('http://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
                 'faster_rcnn_r50_fpn_2x_coco/'
                 'faster_rcnn_r50_fpn_2x_coco_'
                 'bbox_mAP-0.384_20200504_210434-a5d8aa15.pth'),
        help='human detection checkpoint file/url')

    parser.add_argument(
        '--pose-config',
        default='demo/hrnet_w32_coco_256x192.py',
        help='human pose estimation config file path (from mmpose)')

    parser.add_argument(
        '--pose-checkpoint',
        default=('https://download.openmmlab.com/mmpose/top_down/hrnet/'
                 'hrnet_w32_coco_256x192-c78dce93_20200708.pth'),
        help='human pose estimation checkpoint file/url')

    parser.add_argument(
        '--det-score-thr',
        type=float,
        default=0.9,
        help='the threshold of human detection score')

    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    
    parser.add_argument(
        '--clip-len',
        type=int,
        default=48,
        help='specify the clip length of video')

    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    
    # Training config 
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Batch size of training')
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of epochs')
    parser.add_argument(
        '--log',
        default='',
        help='Path to the log file')
    # Learing rate
    parser.add_argument(
        '--lr',
        type=float,
        default=0.001,
        help='Learning rate of optimizer')
    parser.add_argument(
        '--gamma',
        type=float,
        default=0.05,
        help='percent reduce of learning rate per 10 epoch')
    parser.add_argument(
    '--checkpoint-dir',
    default='',
    help='Checkpoint dir to store training checkpoint')
    return parser.parse_args()

class PoseC3DTransform:
    def __init__(self,
                cfg,
                sample_duration: int=48,
                mode: str='test',
                sample_mode: str='sequence',
                num_clips: int=1,
                start_index: int=1,
                modality: str='RGB',
                input_flag: str='video',
                seed: int=-1,
                ram: bool=True,
                ):
        self.cfg = cfg
        self.mode = mode
        self.sample_duration = sample_duration
        self.sample_mode = sample_mode
        self.num_clips = num_clips
        self.start_index = start_index
        self.modality = modality
        self.input_flag = input_flag
        self.seed = seed
        self.ram = ram
        if self.ram:
            self.cache = {}

        self._getPipeline()
        

    def _getPipeline(self):
        if self.sample_mode == 'sequence':
            pipelines = (self.cfg.data.train.pipeline if self.mode=='train' else self.cfg.data.test.pipeline)[1:-3]
        elif self.sample_mode == 'uniform':
            pipelines = (self.cfg.data.train.pipeline if self.mode=='train' else self.cfg.data.test.pipeline)[:-3]
        self._preprocess = Compose(pipelines)
    
    def _toTensor(self, buffer):
        buffer = np.transpose(buffer, (3, 0, 1, 2))
        buffer = np.expand_dims(buffer, axis=0)
        if len(buffer.shape) == 5:
            buffer = np.expand_dims(buffer, axis=0)
        return torch.from_numpy(buffer)

    def _sample(self, buffer):
        ind = np.random.randint(buffer.shape[0]-self.sample_duration, size=1)[0]
        buffer = buffer[ind: ind+self.sample_duration, :, :,:]
        return buffer

    def _load_video(self, video):
        buffer = copy.deepcopy(video)
        if 'num_clips' not in buffer:
            buffer['num_clips'] = self.num_clips
        if 'clip_len' not in buffer:
            buffer['clip_len'] = self.sample_duration * (buffer['total_frames'] // self.sample_duration)
        if 'modality' not in buffer:
            buffer['modality'] = self.modality
        if 'start_index' not in buffer:
            buffer['start_index'] = self.start_index

        return buffer

    def __call__(self, video):

        if not self.ram:
            buffer = self._load_video(video)
            buffer = self._preprocess(buffer)['imgs']
            if self.sample_mode == 'sequence':
                buffer = self._sample(buffer)
            buffer = self._toTensor(buffer)
        else:
            if video['frame_dir'] not in self.cache:
                buffer = self._load_video(video)
                buffer = self._preprocess(buffer)['imgs']
                if self.sample_mode == 'sequence':
                    buffer = self._sample(buffer)
                buffer = self._toTensor(buffer)
                self.cache[video['frame_dir']] = buffer
            else:
                buffer = self.cache[video['frame_dir']]

        return buffer

def print_model(model):
    params = model.to('cpu')
    for k,v in sorted(params.items()):
        print(k, v.shape)
        params[k] = Variable(torch.from_numpy(v), requires_grad=True)
    
    print('\nTotal parameters: ', sum(v.numel() for v in params.values()))