import glob
import re

import os.path as osp

from .bases import BaseImageDataset
import warnings
import json
import cv2
from tqdm import tqdm
import json
import random
import numpy as np
import os

class PRW(BaseImageDataset):
    """Market1501.

    Reference:
        Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.

    URL: `<http://www.liangzheng.org/Project/project_reid.html>`_

    Dataset statistics:
        - identities: 1501 (+1 for background).
        - images: 12936 (train) + 3368 (query) + 15913 (gallery).
    """
    _junk_pids = [0, -1]
    dataset_dir = ''
    dataset_url = 'http://188.138.127.15:81/Datasets/Market-1501-v15.09.15.zip'

    def __init__(self, root='datasets', market1501_500k=False, train_anno=1, **kwargs):

        # root = "/root/person_search/dataset/multi_person"
        self.root = osp.join(root, 'prw')

        # self.root = "/root/person_search/dataset/multi_person/prw"
        self.train_anno = train_anno
        self.pid_container = set()

        self.gallery_id = []

        # train = self.process_dir("train", relabel=True)
        train = self.process_dir_train(relabel=True)
        query = self.process_dir("query", relabel=False)
        gallery = self.process_dir("gallery", relabel=False)

        query = sorted(query)
        gallery = sorted(gallery)

        # print(query)
        # print(len(query))


        self.train = train
        self.query = query
        self.gallery = gallery
        #
        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info_train(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info_gallery(self.gallery)

        print("Dataset statistics:")
        print("  ----------------------------------------")
        print("  subset   | # ids | # images | # cameras")
        print("  ----------------------------------------")
        print(
            "  train    | {:5d} | {:8d} | {:9d}".format(self.num_train_pids, self.num_train_imgs, self.num_train_cams))
        print(
            "  query    | {:5d} | {:8d} | {:9d}".format(self.num_query_pids, self.num_query_imgs, self.num_query_cams))
        print("  gallery  | {:5d} | {:8d} | {:9d}".format(self.num_gallery_pids, self.num_gallery_imgs,
                                                          self.num_gallery_cams))
        print("  ----------------------------------------")


    def get_imagedata_info_train(self, data):

        pids, cams = [], []
        for _, _, pid, camid, pid2, pos_neg in data:
            pids += [pid]
            pids += [pid2]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def get_imagedata_info_gallery(self, data):
        pids, cams = [], []
        for _, pid, camid in data:
            if isinstance(pid, list):
                for one_pid in pid:
                    pids += [one_pid]
            cams += [camid]
        pids = set(pids)
        cams = set(cams)
        num_pids = len(pids)
        num_cams = len(cams)
        num_imgs = len(data)
        return num_pids, num_imgs, num_cams

    def process_dir_train(self, relabel=True):
        # # root = "/root/person_search/dataset/person_search/prw"
        anno_path = osp.join(self.root, "gt_training_box.json")

        with open(anno_path, 'r+') as f:
            all_anno = json.load(f)

        pid_container = set()
        for img_name, pid in all_anno.items():
            pid_container.add(pid)
        pid2label = {int(pid): label for label, pid in enumerate(pid_container)}
        # print(pid2label)


        new_anno_path = osp.join(self.root, "pair_pos_unary" + str(self.train_anno) + ".json")
        with open(new_anno_path, 'r+') as f:
            all_anno = json.load(f)
        data = []

        # img_root1 = "/root/person_search/dataset/multi_person/prw/hard_gallery_train/image"
        # img_root2 = "/root/person_search/dataset/multi_person/prw/train_gt/image"

        img_root1 = os.path.join(self.root, 'hard_gallery_train/image')
        img_root2 = os.path.join(self.root, 'train_gt/image')

        for one_pair in all_anno:
            # print(one_pair)
            hard_imgname = one_pair[0]
            query_train_imgname1 = one_pair[1]
            pid1 = one_pair[2]
            query_train_imgname2 = one_pair[3]
            pid2 = one_pair[4]
            camera_id = one_pair[5]
            if relabel:
                pid1 = pid2label[pid1]
                pid2 = pid2label[pid2]
            hard_imgname_path = osp.join(img_root1, hard_imgname)
            query_train_path1 = osp.join(img_root2, query_train_imgname1)
            query_train_path2 = osp.join(img_root2, query_train_imgname2)
            new_anno = [hard_imgname_path, query_train_path1, pid1, query_train_path2, pid2, camera_id]
            data.append(new_anno)
        return data

    def process_dir(self, dataset, relabel=False):

        if dataset == "query":
            anno_path = osp.join(self.root, "query", "query.json")
            img_root = osp.join(self.root, "query", "query_image")
            camid_path = osp.join(self.root, "query", "camera_id.json")
        elif dataset == "gallery":
            gallery_name = "hard_gallery_test"
            anno_path = osp.join(self.root, gallery_name, "gallery.json")
            img_root = osp.join(self.root, gallery_name, "image")
            camid_path = osp.join(self.root, gallery_name, "camera_id.json")

        with open(anno_path, 'r+') as f:
            all_anno = json.load(f)



        if dataset == "query" or dataset == "gallery":
            with open(camid_path, 'r+') as f:
                camid_dic = json.load(f)

        # valid_pid_path = "/root/person_search/dataset/multi_person/prw/valid_q_pid_3.json"
        valid_pid_path = os.path.join(self.root, 'valid_q_pid.json')

        with open(valid_pid_path, 'r+') as f:
            valid_pid = json.load(f)


        data = []
        pid_set = set()
        for img_name, pid in all_anno.items():
            image_path = osp.join(img_root, img_name)
            if dataset == "query":
                print({img_name: pid})
            if dataset == "query" or dataset == "gallery":
                camid = camid_dic[img_name]
            if isinstance(pid, str):
                pid = int(pid)
            if dataset == "query":
                if pid in valid_pid:
                    if pid in pid_set:
                        continue
                    else:
                        pid_set.add(pid)
                else:
                    continue

            data.append((image_path, pid, int(camid)))

        return data

