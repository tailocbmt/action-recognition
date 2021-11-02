import torch
import pickle
import os.path as osp
import numpy as np
import pandas as pd
from mmcv import Config

from utils.models import modelFromConfig
from utils.mics import parse_args, PoseC3DTransform, createKNN

def main(args):
    cfg = Config.fromfile(args.cfg_path)
    # Build model
    model = modelFromConfig(cfg=cfg, checkpoint=args.checkpoint, device=args.device)
    model.eval()
    # Read file contains path
    pathFile = pd.read_csv(args.txt_path, names=['filename', 'label', 'status'], header=None)
    pathFile = pathFile.replace(r'\\','/', regex=True)
    # Read keypoint annotation
    with open(args.kp_annotation, 'rb') as f:
        kp_annotation = pickle.load(f)

    # Embed dataset
    loader = PoseC3DTransform(cfg=cfg, 
                        seed=255)
    if args.save:
        pathList = pathFile.loc[pathFile['status'] != 'test', :]
        
        embeddings = np.zeros((len(pathList), model.cls_head.lstm.hidden_size))
        count = 0
        with torch.no_grad():
            for index, row in pathList.iterrows():
                path = row['filename']
                
                keypoint_dict = None
                frameDirPath = osp.join('/content',path.split('.')[0])
                for i in range(len(kp_annotation)):
                    if kp_annotation[i]['frame_dir'] == frameDirPath:
                        keypoint_dict = kp_annotation[i]
                        break
                video = loader(keypoint_dict)
                if args.device == 'cuda':
                    video = video.to(args.device)
                pred = model(video, -1)
                embeddings[count] = pred.to('cpu').numpy()
                count += 1
        np.save(args.embed_path, embeddings)
    else:
        if not args.embed_path.endswith('.npy'):
            embed_path = args.embed_path + '.npy'
        embeddings = np.load(embed_path)
    
    knnModel = createKNN(embeddings, 30)
    pathList = pathFile.loc[pathFile['status'] == 'test', :].reset_index()
    
    with torch.no_grad():
        for k in range(1,args.top+1):
            pAtK = 0
            mRR = 0
            for index, row in pathList.iterrows():
                path = row['filename']
                keypoint_dict = None
                frameDirPath = osp.join('/content',path.split('.')[0])
                for i in range(len(kp_annotation)):
                    if kp_annotation[i]['frame_dir'] == frameDirPath:
                        keypoint_dict = kp_annotation[i]
                        break
                video = loader(keypoint_dict)
                if args.device == 'cuda':
                    video = video.to(args.device)
                pred = model(video, -1)
                dists, ids = knnModel.kneighbors(pred.to('cpu').numpy(), k)

                similarLabelList = pathFile.loc[pathFile['label']==row['label']].index
                isIn = np.isin(np.asarray(ids), np.asarray(similarLabelList))
                pAtKHit = np.any(isIn)
                if pAtKHit:
                    pAtK += 1
            print('Precision at {}: {}%'.format(k ,pAtK*100/len(pathList)))
            print('Mean reciprocal rank: {}%'.format(mRR*100/len(pathList)))
if __name__ == "__main__":
    args = parse_args()
    main(args)
