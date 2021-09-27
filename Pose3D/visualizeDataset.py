import argparse
import os
import os.path as osp

import pickle
import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction

from mmaction.utils import import_module_error_func

try:
    from mmpose.apis import (init_pose_model, inference_top_down_pose_model,
                             vis_pose_result)
except (ImportError, ModuleNotFoundError):

    @import_module_error_func('mmdet')
    def inference_detector(*args, **kwargs):
        pass

    @import_module_error_func('mmdet')
    def init_detector(*args, **kwargs):
        pass

    @import_module_error_func('mmpose')
    def init_pose_model(*args, **kwargs):
        pass

    @import_module_error_func('mmpose')
    def inference_top_down_pose_model(*args, **kwargs):
        pass

    @import_module_error_func('mmpose')
    def vis_pose_result(*args, **kwargs):
        pass

try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

def parse_args():
    parser = argparse.ArgumentParser(description='Create Data format POSE3D')
    parser.add_argument('src', help='source directory', default='/content')
    parser.add_argument('annotation', help='File contains annotation')
    parser.add_argument('out_filename', help='output filename')

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
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    with open(args.annotation, 'rb') as f:
        annotations = pickle.load(f)

    sampleVideo = annotations[16]

    videoName = osp.join(args.src, sampleVideo['frame_dir'])
    frame_paths = sorted(os.listdir(videoName))
    pose_results = sampleVideo['keypoint']
    num_frame = sampleVideo['total_frames']

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)
    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.out_filename, remove_temp=True)

if __name__=="__main__":
    main()