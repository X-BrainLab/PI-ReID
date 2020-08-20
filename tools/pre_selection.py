# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import argparse
import os
import sys
from os import mkdir

import torch
from torch.backends import cudnn
import json

sys.path.append('.')
from config import cfg
from data import make_data_loader
from engine.inference import inference
from modeling import build_model_pre
from utils.logger import setup_logger

import torch
import torch.nn as nn
from ignite.engine import Engine

from utils.reid_metric import R1_mAP, pre_selection_index

def create_supervised_evaluator(model, metrics,
                                device=None):
    """
    Factory function for creating an evaluator for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        metrics (dict of str - :class:`ignite.metrics.Metric`): a map of metric names to Metrics
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.
    Returns:
        Engine: an evaluator engine with supervised inference function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _inference(engine, batch):
        model.eval()
        with torch.no_grad():
            data, pids, camids = batch
            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            feat = model(data)
            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine

# def inference(
#         cfg,
#         model,
#         val_loader,
#         num_query
# ):
#     device = cfg.MODEL.DEVICE
#
#     logger = logging.getLogger("reid_baseline.inference")
#     logger.info("Enter inferencing")
#     if cfg.TEST.RE_RANKING == 'no':
#         print("Create evaluator")
#         evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM)},
#                                                 device=device)
#     elif cfg.TEST.RE_RANKING == 'yes':
#         print("Create evaluator for reranking")
#         evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP_reranking(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM)},
#                                                 device=device)
#     else:
#         print("Unsupported re_ranking config. Only support for no or yes, but got {}.".format(cfg.TEST.RE_RANKING))
#
#     evaluator.run(val_loader)
#     cmc, mAP, _ = evaluator.state.metrics['r1_mAP']
#     logger.info('Validation Results')
#     logger.info("mAP: {:.1%}".format(mAP))
#     for r in [1, 5, 10, 100]:
#         logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))



def main():
    parser = argparse.ArgumentParser(description="ReID Baseline Inference")
    parser.add_argument(
        "--config_file", default="", help="path to config file", type=str
    )
    parser.add_argument("opts", help="Modify config options using the command-line", default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1

    if args.config_file != "":
        cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir and not os.path.exists(output_dir):
        mkdir(output_dir)

    if cfg.MODEL.DEVICE == "cuda":
        os.environ['CUDA_VISIBLE_DEVICES'] = cfg.MODEL.DEVICE_ID
    cudnn.benchmark = True

    train_loader, val_loader, num_query, num_classes = make_data_loader(cfg)
    model = build_model_pre(cfg, num_classes)
    model.load_param(cfg.TEST.WEIGHT)

    # inference(cfg, model, val_loader, num_query)
    device = cfg.MODEL.DEVICE

    evaluator = create_supervised_evaluator(model, metrics={
        'pre_selection_index': pre_selection_index(num_query, max_rank=100, feat_norm=cfg.TEST.FEAT_NORM)},
                                            device=device)

    evaluator.run(val_loader)

    index = evaluator.state.metrics['pre_selection_index']

    with open(cfg.Pre_Index_DIR, 'w+') as f:
        json.dump(index.tolist(), f)

    print("Pre_Selection_Done")

if __name__ == '__main__':
    main()
