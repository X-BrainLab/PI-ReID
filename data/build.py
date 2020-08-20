# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

from torch.utils.data import DataLoader

from .collate_batch import train_collate_fn, val_collate_fn, train_collate_fn_pair, val_collate_fn_pair, train_collate_fn_pair3
from .datasets import init_dataset, ImageDataset, ImageDataset_pair, ImageDataset_pair_val, ImageDataset_pair3
from .samplers import RandomIdentitySampler, RandomIdentitySampler_alignedreid  # New add by gu
from .transforms import build_transforms
import json
import numpy as np
import random
import os

import time


def multi_person_training_info_prw(train_anno, root):
    root = os.path.join(root, 'prw')
    path_gt = os.path.join(root, 'each_pid_info.json')

    with open(path_gt, 'r') as f:
        each_pid_info = json.load(f)
    # print(each_pid_info)
    path_hard = os.path.join(root, 'hard_gallery_train/gallery.json')

    with open(path_hard, 'r') as f:
        path_hard = json.load(f)
    # print(path_hard)
    path_hard_camera_id = os.path.join(root, 'hard_gallery_train/camera_id.json')
    with open(path_hard_camera_id, 'r') as f:
        path_hard_camera_id = json.load(f)
    # print(path_hard_camera_id)

    pairs_anno = []
    for img, pids in path_hard.items():
        camera_id = path_hard_camera_id[img]
        if len(pids) < 2:
            continue
        one_pair = [img]
        for index, pid in enumerate(pids):
            pid_info = each_pid_info[str(pid)]
            pid_info_camera_id = np.array(pid_info[0])
            pos_index = np.where(pid_info_camera_id != camera_id)[0]
            if len(pos_index) == 0:
                continue
            query_img = pid_info[1][random.choice(pos_index)]
            one_pair = one_pair + [query_img, pid]

        one_pair = one_pair + [camera_id]
        if len(one_pair) > 5:
            second_pair = [one_pair[0], one_pair[3], one_pair[4], one_pair[1], one_pair[2], one_pair[5]]
            pairs_anno.append(one_pair)
            pairs_anno.append(second_pair)
    # print(len(pairs_anno))
    anno_save_path = os.path.join(root, "pair_pos_unary" + str(train_anno) + ".json")
    with open(anno_save_path, 'w+') as f:
        json.dump(pairs_anno, f)

def multi_person_training_info_cuhk(train_anno, root):
    root = os.path.join(root, 'cuhk')
    path_gt = os.path.join(root, 'each_pid_info.json')
    with open(path_gt, 'r') as f:
        each_pid_info = json.load(f)
    # print(each_pid_info)

    path_hard = os.path.join(root, 'hard_gallery_train/gallery.json')
    with open(path_hard, 'r') as f:
        path_hard = json.load(f)
    # print(path_hard)

    path_hard_camera_id = os.path.join(root, 'hard_gallery_train/camera_id.json')
    with open(path_hard_camera_id, 'r') as f:
        path_hard_camera_id = json.load(f)
    # print(path_hard_camera_id)


    pairs_anno = []
    count2 = 0
    for img, pids in path_hard.items():
        # camera_id = path_hard_camera_id[img]
        if len(pids) < 2:
            continue
        count2+=1
        # else:
        #     continue
        one_pair = [img]
        camera_id = 0
        for index, pid in enumerate(pids):
            pid_info = each_pid_info[str(pid)]
            # pid_info_camera_id = np.array(pid_info[0])
            # pos_index = np.where(pid_info_camera_id != camera_id)[0]
            # if len(pos_index) == 0:
            #     continue
            # query_img = pid_info[1][random.choice(pos_index)]
            query_img = random.choice(pid_info[1])
            one_pair = one_pair + [query_img, pid]

        one_pair = one_pair + [camera_id]
        if len(one_pair) > 5:
            second_pair = [one_pair[0], one_pair[3], one_pair[4], one_pair[1], one_pair[2], one_pair[5]]
            pairs_anno.append(one_pair)
            pairs_anno.append(second_pair)

    anno_save_path = os.path.join(root, "pair_pos_unary" + str(train_anno) + ".json")
    with open(anno_save_path, 'w+') as f:
        json.dump(pairs_anno, f)

def make_data_loader(cfg):
    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    num_classes = dataset.num_train_pids
    train_set = ImageDataset(dataset.train, train_transforms)
    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )
    return train_loader, val_loader, len(dataset.query), num_classes

def make_data_loader_train(cfg):
    # multi_person_training_info2(cfg.DATASETS.TRAIN_ANNO)

    if "cuhk" in cfg.DATASETS.NAMES:
        multi_person_training_info_cuhk(cfg.DATASETS.TRAIN_ANNO, cfg.DATASETS.ROOT_DIR)
    else:
        multi_person_training_info_prw(cfg.DATASETS.TRAIN_ANNO, cfg.DATASETS.ROOT_DIR)

    train_transforms = build_transforms(cfg, is_train=True)
    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS
    if len(cfg.DATASETS.NAMES) == 1:
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, train_anno=cfg.DATASETS.TRAIN_ANNO)
    else:
        # TODO: add multi dataset to train
        dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR, train_anno=cfg.DATASETS.TRAIN_ANNO)

    train_set = ImageDataset_pair3(dataset.train, train_transforms)
    num_classes = dataset.num_train_pids

    if cfg.DATALOADER.SAMPLER == 'softmax':
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH, shuffle=True, num_workers=num_workers,
            collate_fn=train_collate_fn_pair3
        )
    else:
        train_loader = DataLoader(
            train_set, batch_size=cfg.SOLVER.IMS_PER_BATCH,
            sampler=RandomIdentitySampler(dataset.train, cfg.SOLVER.IMS_PER_BATCH, cfg.DATALOADER.NUM_INSTANCE),
            # sampler=RandomIdentitySampler_alignedreid(dataset.train, cfg.DATALOADER.NUM_INSTANCE),      # new add by gu
            num_workers=num_workers, collate_fn=train_collate_fn_pair3
        )

    val_set = ImageDataset(dataset.query + dataset.gallery, val_transforms)
    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn
    )

    return train_loader, val_loader, len(dataset.query), num_classes

def make_data_loader_val(cfg, index, dataset):

    indice_path = cfg.Pre_Index_DIR
    with open(indice_path, 'r') as f:
        indices = json.load(f)
    indice = indices[index][:100]

    val_transforms = build_transforms(cfg, is_train=False)
    num_workers = cfg.DATALOADER.NUM_WORKERS

    # if len(cfg.DATASETS.NAMES) == 1:
    #     dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)
    # else:
    #     # TODO: add multi dataset to train
    #     print(cfg.DATASETS.NAMES)
    #     dataset = init_dataset(cfg.DATASETS.NAMES, root=cfg.DATASETS.ROOT_DIR)

    query = dataset.query[index]
    gallery = [dataset.gallery[ind] for ind in indice]
    gallery = [query] + gallery

    val_set = ImageDataset_pair_val(query, gallery, val_transforms)

    val_loader = DataLoader(
        val_set, batch_size=cfg.TEST.IMS_PER_BATCH, shuffle=False, num_workers=num_workers,
        collate_fn=val_collate_fn_pair
    )

    return val_loader


