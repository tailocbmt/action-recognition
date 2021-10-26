import torch
import numpy as np
import pandas as pd
from mmcv import Config

from utils.models import modelFromConfig
from utils.mics import parse_args, PoseC3DTransform, createKNN

def main(args):
    cfg = Config.fromfile(args.cfgPath)
    # Build model
    model = modelFromConfig(cfg=cfg, device=args.device)
    model.eval()
    # Read file contains path
    pathFile = pd.read_csv(args.txtPath, names=['filename', 'label', 'status'], header=None)
    # Embed dataset
    loader = PoseC3DTransform(cfg_path=args.cfgPath, 
                        kp_annotation=args.kpAnnotation,
                        seed=255)
    if args.save:
        pathList = pathFile.loc[pathFile['status'] != 'test', 'filename']

        embeddings = np.zeros((len(pathList), model.cls_head.fc_cls.out_features))
        with torch.no_grad():
            for i in range(len(pathList)):
                path = pathList[i]
                video = loader(path)
                if args.device == 'cuda':
                    video = video.to(args.device)
                pred = model(video, -1)
                embeddings[i] = pred.to('cpu').numpy()
        np.save(args.embedPath, embeddings)
    else:
        if not args.embedPath.endswith('.npy'):
            embedPath = args.embedPath + '.npy'
        embeddings = np.load(embedPath)
    
    knnModel = createKNN(embeddings, args.top)
    pathList = pathFile.loc[pathFile['status'] == 'test', :].reset_index()
    pAtK = 0
    mRR = 0
    with torch.no_grad():
        for index, row in pathList.iterrows():
            path = row['filename']
            video = loader(path)
            if args.device == 'cuda':
                video = video.to(args.device)
            pred = model(video, -1)
            dists, ids = knnModel.kneighbors(pred.to('cpu').numpy(), args.top)

            similarLabelList = pathFile.loc[pathFile['label']==row['label']].index
            isIn = np.any(np.isin(np.asarray(ids), np.asarray(similarLabelList)))
            if isIn:
                pAtK += 1
    print('Precision at {}: {}%'.format(args.top ,pAtK*100/len(pathList)))
    print('Mean reciprocal rank: {}%'.format(args.top ,mRR))
if __name__ == "__main__":
    args = parse_args()
    main(args)
