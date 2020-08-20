# encoding: utf-8
"""
@author:  liaoxingyu
@contact: sherlockliao01@gmail.com
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not osp.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img_path, pid, camid = self.dataset[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, pid, camid, img_path

class ImageDataset_pair(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        query_path, gallery_path, pid, camid, pid2, pos_neg = self.dataset[index]
        query_img = read_image(query_path)
        gallery_img = read_image(gallery_path)

        if self.transform is not None:
            query_img = self.transform(query_img)
            gallery_img = self.transform(gallery_img)

        return query_img, gallery_img, pid, camid, query_path, gallery_path, pid2, pos_neg

class ImageDataset_pair3(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, dataset, transform=None):
        self.dataset = dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        gallery_path, query_path1, pid1, query_path2, pid2, camera_id = self.dataset[index]
        query_img1 = read_image(query_path1)
        query_img2 = read_image(query_path2)
        gallery_img = read_image(gallery_path)

        if self.transform is not None:
            query_img1 = self.transform(query_img1)
            query_img2 = self.transform(query_img2)
            gallery_img = self.transform(gallery_img)

        return gallery_img, query_img1, pid1, query_img2, pid2, camera_id


class ImageDataset_pair_val(Dataset):
    """Image Person ReID Dataset"""

    def __init__(self, query, gallery, transform=None):
        self.query = query
        self.gallery = gallery
        self.transform = transform

    def __len__(self):
        return len(self.gallery)

    def __getitem__(self, index):

        query_path, pid, camid = self.query
        gallery_path, pid, camid = self.gallery[index]

        is_first = query_path == gallery_path

        query_img = read_image(query_path)
        gallery_img = read_image(gallery_path)

        if self.transform is not None:
            query_img = self.transform(query_img)
            gallery_img = self.transform(gallery_img)

        return query_img, gallery_img, pid, camid, query_path, gallery_path, is_first
