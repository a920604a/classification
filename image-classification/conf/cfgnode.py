'''
Author: yuan
Date: 2021-02-25 14:29:52
LastEditTime: 2021-02-25 14:45:54
FilePath: /aidc-algorithm/image-classification/conf/cfgnode.py
'''
from yacs.config import CfgNode
import torch


def get_cfg(model) -> CfgNode:

    _C = CfgNode()
    _C.IMAGE_SIZE = model.image_size

    _C.MODEL_FILE = model.model_file
    _C.LABLE = model.labels
    # _C.focus_labels = model.focus_labels
    _C.NET = model.net
    _C.GPU = torch.cuda.is_available()
    return _C.clone()
