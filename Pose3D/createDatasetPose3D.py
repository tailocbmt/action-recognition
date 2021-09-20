# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
import os.path as osp

import pickle
import cv2
import mmcv
import numpy as np
import torch
from mmcv import DictAction
from mmcv.runner import load_checkpoint

from mmaction.datasets.pipelines import Compose
from mmaction.models import build_model
from mmaction.utils import import_module_error_func

try:
    from mmdet.apis import inference_detector, init_detector
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


def parse_args():
    parser = argparse.ArgumentParser(description='Create Data format POSE3D')
    parser.add_argument('src', help='source directory', default='/content')
    parser.add_argument('annotation', help='File contains annotation')
    parser.add_argument('savePath', help='Path to save annotation')

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
        '--device', type=str, default='cuda:0', help='CPU/CUDA device option')
    parser.add_argument(
        '--short-side',
        type=int,
        default=480,
        help='specify the short-side length of the image')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        default={},
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. For example, '
        "'--cfg-options model.backbone.depth=18 model.backbone.with_cp=True'")
    args = parser.parse_args()
    return args


def frame_extraction(src , annotationPath, short_side):
    """Extract frames given video_path.
    Args:
        video_path (str): The video_path.
    """
    videoPaths = open(annotationPath, 'r')
    video_paths = []
    videoLabels = []
    frameHW = None

    for line in videoPaths.readlines():
        # Load the video, extract frames into ./tmp/video_name
        line = line.split()
        videoLabels.append(line[1])
        
        videoPath = osp.join(src, line[0])
        target_dir = videoPath.split('.')[0]
        os.makedirs(target_dir, exist_ok=True)
        # Should be able to handle videos up to several hours
        frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
        vid = cv2.VideoCapture(videoPath)
        
        video_paths.append(target_dir)
        flag, frame = vid.read()
        cnt = 0
        new_h, new_w = None, None
        while flag:
            if new_h is None:
                h, w, _ = frame.shape
                new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

            frame = mmcv.imresize(frame, (new_w, new_h))

            frameHW = (new_w, new_h)
            frame_path = frame_tmpl.format(cnt + 1)
            

            cv2.imwrite(frame_path, frame)
            cnt += 1
            flag, frame = vid.read()

    return video_paths, frameHW, videoLabels


def detection_inference(args, video_paths):
    """Detect human boxes given frame paths.
    Args:
        args (argparse.Namespace): The arguments.
        frame_paths (list[str]): The paths of frames to do detection inference.
    Returns:
        list[np.ndarray]: The human detection results.
    """
    model = init_detector(args.det_config, args.det_checkpoint, args.device)
    assert model.CLASSES[0] == 'person', ('We require you to use a detector '
                                          'trained on COCO')
    results = []
    print('Performing Human Detection for each frame')
    prog_bar = mmcv.ProgressBar(len(video_paths))

    for videoPath in video_paths:
        frameResult = []
        frame_paths = sorted(os.listdir(videoPath))
        for frame_path in frame_paths:
            result = inference_detector(model, osp.join(videoPath,frame_path))
            # We only keep human detections with score larger than det_score_thr
            result = result[0][result[0][:, 4] >= args.det_score_thr]
            frameResult.append(result)
        
        results.append(frameResult)
        prog_bar.update()
    return results


def pose_inference(args, video_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)

    results = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(video_paths))

    for videoPath in video_paths:
        frameResult = []
        frame_paths = sorted(os.listdir(videoPath))
        for f, d in zip(frame_paths, det_results):
            # Align input format
            d = [dict(bbox=x) for x in list(d)]
            pose = inference_top_down_pose_model(model, osp.join(videoPath,f), d, format='xyxy')[0]
            frameResult.append(pose)
        
        results.append(frameResult)
        prog_bar.update()
    return results

def createAnnotation(pose_results, frameHW, video_paths, videoLabels):
    # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
    h,w = frameHW
    num_keypoint = 17
    
    fake_anno = dict(
        frame_dir='',
        label=-1,
        img_shape=(h, w),
        original_shape=(h, w),
        total_frames=0)

    annotations = []

    for i in range(len(pose_results)):
        frame_dir = video_paths[i]
        num_frame = len(os.listdir(frame_dir))
        label = videoLabels[i]

        num_person = max([len(x) for x in pose_results[i]])

        keypoint = np.zeros((num_person, num_frame, num_keypoint, 2),
                        dtype=np.float16)
        keypoint_score = np.zeros((num_person, num_frame, num_keypoint),
                                dtype=np.float16)

        for j, poses in enumerate(pose_results[i]):
            for k, pose in enumerate(poses):
                pose = pose['keypoints']
                keypoint[k, j] = pose[:, :2]
                keypoint_score[k, j] = pose[:, 2]

        fake_anno['frame_dir'] = frame_dir
        fake_anno['label'] = label
        fake_anno['total_frames'] = num_frame
        fake_anno['keypoint'] = keypoint
        fake_anno['keypoint_score'] = keypoint_score

        annotations.append(fake_anno)
    return annotations

    
    

def main():
    args = parse_args()

    video_paths, frameHW, videoLabels = frame_extraction(args.src,
                                                        args.annotation,
                                                        args.short_side)


    # Get Human detection results
    det_results = detection_inference(args, video_paths)
    torch.cuda.empty_cache()

    pose_results = pose_inference(args, video_paths, det_results)
    torch.cuda.empty_cache()

    annotations = createAnnotation(pose_results, frameHW, video_paths, videoLabels)
    
    #Save annotations
    with open(args.savePath, 'wb') as f:
        pickle.dump(annotations, f)

if __name__ == '__main__':
    main()