# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import logging

import torch
import torch.nn as nn
from ignite.engine import Engine, Events
from ignite.handlers import ModelCheckpoint, Timer
from ignite.metrics import RunningAverage
from data import make_data_loader_val

from utils.reid_metric import R1_mAP, R1_mAP_pair
from tqdm import tqdm
import time
import json
import numpy as np
from layers.triplet_loss import TripletLoss

from data.datasets import init_dataset

import copy

import torch.nn.functional as F

from data import make_data_loader, make_data_loader_train
import random
import os

global ITER
ITER = 0


def euclidean_dist(gallery_feature1, gallery_feature2):

    xx = torch.pow(gallery_feature1, 2).sum(1, keepdim=True)
    yy = torch.pow(gallery_feature2, 2).sum(1, keepdim=True)
    dist1 = xx + yy
    dist2 = gallery_feature1 * gallery_feature2
    dist2 = dist2.sum(1, keepdim=True)
    dist = dist1 - 2 * dist2
    dist = dist.clamp(min=1e-12).sqrt()
    return dist

def loss1(gallery_feature1, gallery_feature2, query_feature, margin=0.3):

    ranking_loss = nn.MarginRankingLoss(margin=margin)
    y = gallery_feature1.new((gallery_feature1.shape[0], 1)).fill_(1)
    dist_neg = euclidean_dist(gallery_feature1, gallery_feature2)
    dist_pos = euclidean_dist(gallery_feature1, query_feature)
    loss = ranking_loss(dist_neg, dist_pos, y)

    return loss


def create_supervised_trainer(model, optimizer, loss_fn,
                              device=None, gamma=1.0, margin=0.3, beta=1.0):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        # guiding, img, target, target2, pos_neg = batch

        img, guiding1, guiding2, target1, target2 = batch

        img = img.to(device) if torch.cuda.device_count() >= 1 else img

        guiding1 = guiding1.to(device) if torch.cuda.device_count() >= 1 else guiding1
        target1 = target1.to(device) if torch.cuda.device_count() >= 1 else target1

        guiding2 = guiding2.to(device) if torch.cuda.device_count() >= 1 else guiding2
        target2 = target2.to(device) if torch.cuda.device_count() >= 1 else target2

        # score, feat, score_guiding, feature_guiding, gallery_attention, score_pos_neg = model(guiding1, img, x_g2=guiding2)
        score, feat, score1, feat1, feat_query, score2, feat2 = model(guiding1, img, x_g2=guiding2)

        loss = loss_fn(score, feat, target1) + gamma * loss1(feat, feat1.detach(), feat_query, margin=margin) + beta * loss_fn(score2, feat, target1)

        loss.backward()
        optimizer.step()

        acc = (score.max(1)[1] == target1).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


def create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cetner_loss_weight,
                              device=None):
    """
    Factory function for creating a trainer for supervised models

    Args:
        model (`torch.nn.Module`): the model to train
        optimizer (`torch.optim.Optimizer`): the optimizer to use
        loss_fn (torch.nn loss function): the loss function to use
        device (str, optional): device type specification (default: None).
            Applies to both model and batches.

    Returns:
        Engine: a trainer engine with supervised update function
    """
    if device:
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        model.to(device)

    def _update(engine, batch):
        model.train()
        optimizer.zero_grad()
        optimizer_center.zero_grad()
        img, target = batch
        img = img.to(device) if torch.cuda.device_count() >= 1 else img
        target = target.to(device) if torch.cuda.device_count() >= 1 else target
        score, feat = model(img)
        loss = loss_fn(score, feat, target)
        # print("Total loss is {}, center loss is {}".format(loss, center_criterion(feat, target)))
        loss.backward()
        optimizer.step()
        for param in center_criterion.parameters():
            param.grad.data *= (1. / cetner_loss_weight)
        optimizer_center.step()

        # compute acc
        acc = (score.max(1)[1] == target).float().mean()
        return loss.item(), acc.item()

    return Engine(_update)


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
            guiding, data, pids, camids, is_first = batch

            data = data.to(device) if torch.cuda.device_count() >= 1 else data
            guiding = guiding.to(device) if torch.cuda.device_count() >= 1 else guiding
            feat = model(guiding, data, is_first=is_first)

            return feat, pids, camids

    engine = Engine(_inference)

    for name, metric in metrics.items():
        metric.attach(engine, name)

    return engine


def do_train(
        cfg,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer(model, optimizer, loss_fn, device=device, gamma=cfg.MODEL.GAMMA, margin=cfg.SOLVER.MARGIN, beta=cfg.MODEL.BETA)
    if cfg.TEST.PAIR == "no":
        evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(1, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    elif cfg.TEST.PAIR == "yes":
        evaluator = create_supervised_evaluator(model, metrics={
            'r1_mAP': R1_mAP_pair(1, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    # checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer})
    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1
        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        # multi_person_training_info2()
        train_loader, val_loader, num_query, num_classes = make_data_loader_train(cfg)
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        # if engine.state.epoch % eval_period == 0:
        if engine.state.epoch >= eval_period:
            all_cmc = []
            all_AP = []
            num_valid_q = 0
            q_pids = []
            for query_index in tqdm(range(num_query)):

                val_loader = make_data_loader_val(cfg, query_index, dataset)
                evaluator.run(val_loader)
                cmc, AP, q_pid = evaluator.state.metrics['r1_mAP']

                if AP >= 0:
                    if cmc.shape[0] < 50:
                        continue
                    num_valid_q += 1

                    all_cmc.append(cmc)
                    all_AP.append(AP)
                    q_pids.append(int(q_pid))
                else:
                    continue

            all_cmc = np.asarray(all_cmc).astype(np.float32)
            cmc = all_cmc.sum(0) / num_valid_q
            mAP = np.mean(all_AP)
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))


    trainer.run(train_loader, max_epochs=epochs)


def do_train_with_center(
        cfg,
        model,
        center_criterion,
        train_loader,
        val_loader,
        optimizer,
        optimizer_center,
        scheduler,
        loss_fn,
        num_query,
        start_epoch
):
    log_period = cfg.SOLVER.LOG_PERIOD
    checkpoint_period = cfg.SOLVER.CHECKPOINT_PERIOD
    eval_period = cfg.SOLVER.EVAL_PERIOD
    output_dir = cfg.OUTPUT_DIR
    device = cfg.MODEL.DEVICE
    epochs = cfg.SOLVER.MAX_EPOCHS

    logger = logging.getLogger("reid_baseline.train")
    logger.info("Start training")
    trainer = create_supervised_trainer_with_center(model, center_criterion, optimizer, optimizer_center, loss_fn, cfg.SOLVER.CENTER_LOSS_WEIGHT, device=device)
    evaluator = create_supervised_evaluator(model, metrics={'r1_mAP': R1_mAP(num_query, max_rank=50, feat_norm=cfg.TEST.FEAT_NORM)}, device=device)
    checkpointer = ModelCheckpoint(output_dir, cfg.MODEL.NAME, checkpoint_period, n_saved=10, require_empty=False)
    timer = Timer(average=True)

    trainer.add_event_handler(Events.EPOCH_COMPLETED, checkpointer, {'model': model,
                                                                     'optimizer': optimizer,
                                                                     'center_param': center_criterion,
                                                                     'optimizer_center': optimizer_center})

    timer.attach(trainer, start=Events.EPOCH_STARTED, resume=Events.ITERATION_STARTED,
                 pause=Events.ITERATION_COMPLETED, step=Events.ITERATION_COMPLETED)

    # average metric to attach on trainer
    RunningAverage(output_transform=lambda x: x[0]).attach(trainer, 'avg_loss')
    RunningAverage(output_transform=lambda x: x[1]).attach(trainer, 'avg_acc')

    @trainer.on(Events.STARTED)
    def start_training(engine):
        engine.state.epoch = start_epoch

    @trainer.on(Events.EPOCH_STARTED)
    def adjust_learning_rate(engine):
        scheduler.step()

    @trainer.on(Events.ITERATION_COMPLETED)
    def log_training_loss(engine):
        global ITER
        ITER += 1

        if ITER % log_period == 0:
            logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                        .format(engine.state.epoch, ITER, len(train_loader),
                                engine.state.metrics['avg_loss'], engine.state.metrics['avg_acc'],
                                scheduler.get_lr()[0]))
        if len(train_loader) == ITER:
            ITER = 0

    # adding handlers using `trainer.on` decorator API
    @trainer.on(Events.EPOCH_COMPLETED)
    def print_times(engine):
        logger.info('Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]'
                    .format(engine.state.epoch, timer.value() * timer.step_count,
                            train_loader.batch_size / timer.value()))
        logger.info('-' * 10)
        timer.reset()

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(engine):
        if engine.state.epoch % eval_period == 0:
            evaluator.run(val_loader)
            cmc, mAP = evaluator.state.metrics['r1_mAP']
            logger.info("Validation Results - Epoch: {}".format(engine.state.epoch))
            logger.info("mAP: {:.1%}".format(mAP))
            for r in [1, 5, 10]:
                logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))

    trainer.run(train_loader, max_epochs=epochs)
