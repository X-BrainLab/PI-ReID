# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import torch


def train_collate_fn(batch):
    imgs, pids, _, _, = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids

def train_collate_fn_pair(batch):
    imgs_query, img_gallery, pids, _, _ , _, pids2, pos_neg = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    pids2 = torch.tensor(pids2, dtype=torch.int64)
    pos_neg = torch.FloatTensor(pos_neg)
    # pos_neg = torch.tensor(pos_neg)

    return torch.stack(imgs_query, dim=0), torch.stack(img_gallery, dim=0), pids, pids2, pos_neg

def train_collate_fn_pair3(batch):
    img_gallery, imgs_query1, pids1, imgs_query2, pids2, _ = zip(*batch)
    pids1 = torch.tensor(pids1, dtype=torch.int64)
    pids2 = torch.tensor(pids2, dtype=torch.int64)
    return torch.stack(img_gallery, dim=0), torch.stack(imgs_query1, dim=0), torch.stack(imgs_query2, dim=0), pids1, pids2

def val_collate_fn(batch):
    imgs, pids, camids, _ = zip(*batch)
    return torch.stack(imgs, dim=0), pids, camids

def val_collate_fn_pair(batch):
    imgs_query, imgs_gallery, pids, camids, _ , _, is_first = zip(*batch)
    return torch.stack(imgs_query, dim=0), torch.stack(imgs_gallery, dim=0), pids, camids, is_first
