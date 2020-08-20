# encoding: utf-8
"""
@author:  sherlock
@contact: sherlockliao01@gmail.com
"""

import glob
import re

import os.path as osp

from .bases import BaseImageDataset
import json


class Market1501(BaseImageDataset):
    """
    Market1501
    Reference:
    Zheng et al. Scalable Person Re-identification: A Benchmark. ICCV 2015.
    URL: http://www.liangzheng.org/Project/project_reid.html

    Dataset statistics:
    # identities: 1501 (+1 for background)
    # images: 12936 (train) + 3368 (query) + 15913 (gallery)
    """
    dataset_dir = 'market1501'

    def __init__(self, root='/home/haoluo/data', train_anno=1, verbose=True, **kwargs):
        super(Market1501, self).__init__()
        self.dataset_dir = osp.join(root, self.dataset_dir)
        self.train_dir = osp.join(self.dataset_dir, 'bounding_box_train')
        self.query_dir = osp.join(self.dataset_dir, 'query')
        self.gallery_dir = osp.join(self.dataset_dir, 'bounding_box_test')

        # self._check_before_run()

        self.root = "/raid/home/henrayzhao/person_search/dataset/multi_person/prw"
        # self.multi_person_training_info2()
        self.train_anno = train_anno


        train = self.process_dir_train(relabel=True)
        query = self._process_dir(self.query_dir, relabel=False)
        gallery = self._process_dir(self.gallery_dir, relabel=False)

        # if verbose:
        #     print("=> Market1501 loaded")
        #     self.print_dataset_statistics(train, query, gallery)

        self.train = train
        self.query = query
        self.gallery = gallery

        self.num_train_pids, self.num_train_imgs, self.num_train_cams = self.get_imagedata_info_train(self.train)
        self.num_query_pids, self.num_query_imgs, self.num_query_cams = self.get_imagedata_info(self.query)
        self.num_gallery_pids, self.num_gallery_imgs, self.num_gallery_cams = self.get_imagedata_info(self.gallery)
    #
    # def _check_before_run(self):
    #     """Check if all files are available before going deeper"""
    #     if not osp.exists(self.dataset_dir):
    #         raise RuntimeError("'{}' is not available".format(self.dataset_dir))
    #     if not osp.exists(self.train_dir):
    #         raise RuntimeError("'{}' is not available".format(self.train_dir))
    #     if not osp.exists(self.query_dir):
    #         raise RuntimeError("'{}' is not available".format(self.query_dir))
    #     if not osp.exists(self.gallery_dir):
    #         raise RuntimeError("'{}' is not available".format(self.gallery_dir))


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

    def _process_dir(self, dir_path, relabel=False):
        img_paths = glob.glob(osp.join(dir_path, '*.jpg'))
        pattern = re.compile(r'([-\d]+)_c(\d)')

        pid_container = set()
        for img_path in img_paths:
            pid, _ = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            pid_container.add(pid)
        pid2label = {pid: label for label, pid in enumerate(pid_container)}

        dataset = []
        for img_path in img_paths:
            pid, camid = map(int, pattern.search(img_path).groups())
            if pid == -1: continue  # junk images are just ignored
            assert 0 <= pid <= 1501  # pid == 0 means background
            assert 1 <= camid <= 6
            camid -= 1  # index starts from 0
            if relabel: pid = pid2label[pid]
            dataset.append((img_path, pid, camid))
            if 'query' in dir_path and len(dataset) >= 300:
                break

        return dataset

    def process_dir_train(self, relabel=True):
        root = "/raid/home/henrayzhao/person_search/dataset/person_search/prw"
        anno_path = osp.join(root, "training_box", "training_box.json")
        with open(anno_path, 'r+') as f:
            all_anno = json.load(f)

        pid_container = set()
        for img_name, pid in all_anno.items():
            pid_container.add(pid)
        pid2label = {int(pid): label for label, pid in enumerate(pid_container)}

        new_anno_path = osp.join(self.root, "pair_pos_unary" + str(self.train_anno) + ".json")
        with open(new_anno_path, 'r+') as f:
            all_anno = json.load(f)
        data = []

        img_root1 = "/raid/home/henrayzhao/person_search/dataset/multi_person/prw/hard_gallery_train/image"
        img_root2 = "/raid/home/henrayzhao/person_search/dataset/multi_person/prw/train_gt/image"

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
