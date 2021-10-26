import os
from mmaction.models import build_model
from mmcv.runner import load_checkpoint

def modelFromConfig(cfg,
                    checkpoint: str='',
                    pretrained: bool=True,
                    device: str='cuda'):
    """
    Function to build model from config
    Args:
        cfg: An object contained config of model, checkpoint path,...
        pretrained: bool. Whether to load pretrained or not
        device: device to transfer model
    Return:
        model build from config
    """
    model = build_model(cfg=cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model = build_model(cfg=cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if os.path.exists(checkpoint):
        model = model.load_state_dict(checkpoint['state_dict'])
    else:
        if cfg.load_from is not None and pretrained:
            load_checkpoint(model, cfg.load_from, map_location='cuda')
    
    model = model.to(device)
    print(model)
    return model
        