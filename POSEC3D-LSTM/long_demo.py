from collections import deque
import os
import os.path as osp
import shutil

import cv2
import mmcv
import pandas as pd
import numpy as np
import torch
from utils.mics import parse_args, PoseC3DTransform, createKNN
from utils.models import modelFromConfig

from mmaction.apis import init_recognizer
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


try:
    import moviepy.editor as mpy
except ImportError:
    raise ImportError('Please install moviepy to enable output file')

FONTFACE = cv2.FONT_HERSHEY_DUPLEX
FONTSCALE = 0.75
FONTCOLOR = (255, 0, 0)  # BGR, white
THICKNESS = 1
LINETYPE = 1

def frame_extraction(video_path, short_side):
    """Extract frames given video_path.
    Args:
        video_path (str): The video_path.
    """
    # Load the video, extract frames into ./tmp/video_name
    target_dir = osp.join('./tmp', osp.basename(osp.splitext(video_path)[0]))
    os.makedirs(target_dir, exist_ok=True)
    # Should be able to handle videos up to several hours
    frame_tmpl = osp.join(target_dir, 'img_{:06d}.jpg')
    vid = cv2.VideoCapture(video_path)
    frames = []
    frame_paths = []
    flag, frame = vid.read()
    cnt = 0
    new_h, new_w = None, None
    while flag:
        if new_h is None:
            h, w, _ = frame.shape
            new_w, new_h = mmcv.rescale_size((w, h), (short_side, np.Inf))

        frame = mmcv.imresize(frame, (new_w, new_h))

        frames.append(frame)
        frame_path = frame_tmpl.format(cnt + 1)
        frame_paths.append(frame_path)

        cv2.imwrite(frame_path, frame)
        cnt += 1
        flag, frame = vid.read()

    return frame_paths, frames

def detection_inference(args, frame_paths):
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
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for frame_path in frame_paths:
        result = inference_detector(model, frame_path)
        # We only keep human detections with score larger than det_score_thr
        result = result[0][result[0][:, 4] >= args.det_score_thr]
        results.append(result)
        prog_bar.update()
    return results


def pose_inference(args, frame_paths, det_results):
    model = init_pose_model(args.pose_config, args.pose_checkpoint,
                            args.device)
    ret = []
    print('Performing Human Pose Estimation for each frame')
    prog_bar = mmcv.ProgressBar(len(frame_paths))
    for f, d in zip(frame_paths, det_results):
        # Align input format
        d = [dict(bbox=x) for x in list(d)]
        pose = inference_top_down_pose_model(model, f, d, format='xyxy')[0]
        ret.append(pose)
        prog_bar.update()
    return ret

def action_inference(args, config, pose_results, shape):
    # Preprocess data function
    loader = PoseC3DTransform(cfg=config, 
                        seed=255,
                        ram=False)

    # Load PoseC3D model
    model = modelFromConfig(cfg=config, checkpoint=args.checkpoint, device=args.device)
    model.eval()

    # Load embeddings
    if not args.embed_path.endswith('.npy'):
        embed_path = args.embed_path + '.npy'
    embeddings = np.load(embed_path)

    # Load K Nearest Neighbors model
    knnModel = createKNN(embeddings, 30)

    # Create class mapping
    pathFile = pd.read_csv(args.txt_path, names=['filename', 'label', 'status'], header=None)
    pathFile = pathFile.replace(r'\\','/', regex=True)
    id2class = dict(
        zip(pathFile.index.tolist(), pathFile['label'].tolist()) 
    )
    # Create fake annotation for model input
    h, w, _ = shape

    num_person = max([len(x) for x in pose_results])
    # Current PoseC3D models are trained on COCO-keypoints (17 keypoints)
    num_keypoint = 17
    keypoint = np.zeros((num_person, args.clip_len, num_keypoint, 2),
                        dtype=np.float16)
    keypoint_score = np.zeros((num_person, args.clip_len, num_keypoint),
                              dtype=np.float16)
    
    
    count = 0
    video_classes = []
    
    for i, poses in enumerate(pose_results):
        for j, pose in enumerate(poses):
            pose = pose['keypoints']
            keypoint[j, count] = pose[:, :2]
            keypoint_score[j, count] = pose[:, 2]
        count+=1    
        
        if (i+1) % len(pose_results) == 0 and count < args.clip_len:
            while count < args.clip_len:
                for j, pose in enumerate(poses):
                    pose = pose['keypoints']
                    keypoint[j, count] = pose[:, :2]
                    keypoint_score[j, count] = pose[:, 2]
                count+=1

        if count == args.clip_len:
            fake_anno = dict(
            frame_dir='',
            label=-1,
            img_shape=(h, w),
            original_shape=(h, w),
            start_index=0,
            modality='Pose',
            total_frames=args.clip_len,
            keypoint=keypoint,
            keypoint_score=keypoint_score)
            
            keypoint = np.zeros((num_person, args.clip_len, num_keypoint, 2),
                        dtype=np.float16)
            keypoint_score = np.zeros((num_person, args.clip_len, num_keypoint),
                                    dtype=np.float16)
            # Predict
            video = loader(fake_anno)
            video = video.to(args.device)
            pred = model(video, -1)
            _, ids = knnModel.kneighbors(pred.to('cpu').detach().numpy(), args.top)

            classes = [id2class[id] for id in ids[0]]

            average_class = classes[0]
            video_classes.append(average_class)
            count = 0
    
    return video_classes




def main():
    args = parse_args()

    frame_paths, original_frames = frame_extraction(args.video,
                                                    args.short_side)
    num_frame = len(frame_paths)

    # Get clip_len, frame_interval and calculate center index of each clip
    config = mmcv.Config.fromfile(args.cfgPath)
    config.merge_from_dict(args.cfg_options)


    # Get Human detection results
    det_results = detection_inference(args, frame_paths)
    torch.cuda.empty_cache()

    pose_results = pose_inference(args, frame_paths, det_results)
    torch.cuda.empty_cache()


    results = action_inference(args, config, pose_results, original_frames[0].shape)
    
    action_label = [value for value in results for _ in range(args.clip_len)]

    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint,
                                 args.device)
    vis_frames = [
        vis_pose_result(pose_model, frame_paths[i], pose_results[i])
        for i in range(num_frame)
    ]
    for i in range(len(vis_frames)):
        cv2.putText(vis_frames[i], ('1' if action_label[i] > 0.5 else '0'), (10, 30), FONTFACE, FONTSCALE,
                    FONTCOLOR, THICKNESS, LINETYPE)

    vid = mpy.ImageSequenceClip([x[:, :, ::-1] for x in vis_frames], fps=24)
    vid.write_videofile(args.outFilename, remove_temp=True)

    tmp_frame_dir = osp.dirname(frame_paths[0])
    shutil.rmtree(tmp_frame_dir)


if __name__ == '__main__':
    main()