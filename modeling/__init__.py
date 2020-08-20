# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

from .PISNet import PISNet
from .Pre_Selection_Model import Pre_Selection_Model


def build_model(cfg, num_classes):
    model = PISNet(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE, has_non_local=cfg.MODEL.HAS_NON_LOCAL, sia_reg=cfg.MODEL.SIA_REG, pyramid=cfg.MODEL.PYRAMID, test_pair=cfg.TEST.PAIR)
    return model

def build_model_pre(cfg, num_classes):
    model = Pre_Selection_Model(num_classes, cfg.MODEL.LAST_STRIDE, cfg.MODEL.PRETRAIN_PATH, cfg.MODEL.NECK, cfg.TEST.NECK_FEAT, cfg.MODEL.NAME, cfg.MODEL.PRETRAIN_CHOICE)
    return model

