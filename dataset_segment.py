import os
import pickle
from collections import defaultdict

import cv2
import numpy as np
import torch
from albumentations import *
from torch.utils.data import Dataset

from dataset import MEAN, SD, cv2_loader, make_transform

here = os.path.dirname(__file__)


class ChestXRay14Segment:
    def __init__(
            self,
            images_path,
            eval_transform,
    ):
        self.segment_only_data = ChestXRay14SegmentPKL(
            images_path,
            f'{here}/data/data_segmentonly.pkl',
            transform=eval_transform,
            with_segment=True,
        )


class ChestXRay14SegmentPKL(Dataset):
    """dataset with segmentaion masks"""
    def __init__(self, dirname, pkl, transform, with_segment=True):
        with open(pkl, 'rb') as f:
            data = pickle.load(f)
        self.dirname = dirname
        self.with_segment = with_segment
        self.names = data['names']
        self.labels = data['labels']
        self.bboxs = data['bboxs']
        self.transform = transform

        self.l_to_i = {
            'Atelectasis': 0,
            'Cardiomegaly': 1,
            'Effusion': 2,
            'Infiltration': 3,
            'Mass': 4,
            'Nodule': 5,
            'Pneumonia': 6,
            'Pneumothorax': 7,
            'Consolidation': 8,
            'Edema': 9,
            'Emphysema': 10,
            'Fibrosis': 11,
            'Pleural_Thickening': 12,
            'Hernia': 13,
        }
        self.i_to_l = {v: k for k, v in self.l_to_i.items()}

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        img_path = os.path.join(self.dirname, name)
        img = cv2_loader(img_path)
        if name in self.bboxs:
            bboxs = self.bboxs[name]['bbox']
            cats = self.bboxs[name]['cat']
        else:
            # no annotation
            bboxs = []
            cats = []
        if self.with_segment:
            seg, has = _make_seg_mask(bboxs, cats)
            has = torch.tensor(has)
        if self.transform is not None:
            if self.with_segment and name in self.bboxs:
                res = self.transform(image=img, mask=seg)
                img = res['image']
                seg = res['mask']
            else:
                res = self.transform(image=img)
                img = res['image']
                seg = None
        y = torch.tensor(self.labels[i]).float()
        if self.with_segment:
            return {
                'img': img,
                'evidence': y,
                'seg': seg,
                'has': has,
            }
        else:
            return {'img': img, 'evidence': y}


def _make_seg_mask(bboxs, cats, w=1024, h=1024, n_cls=14):
    """
    Returns:
        seg: (w, h, has_cats)
        has: (n_cls,)
    """
    uniq = sorted(set(cats))
    cat_to_i = {cat: i for i, cat in enumerate(uniq)}
    seg = np.zeros((w, h, len(uniq)), dtype=np.uint8)
    has = [False] * n_cls
    for (x, y, w, h), cat in zip(bboxs, cats):
        seg[y:y + h, x:x + w, cat_to_i[cat]] = 1
        has[cat] = True
    return seg, has


def collate_fn(items):
    """segment is a list used instead of default collate function"""
    out = defaultdict(list)
    list_keys = ['seg']
    for each in items:
        for k, v in each.items():
            out[k].append(v)
    for k, v in out.items():
        if k not in list_keys:
            out[k] = torch.stack(v)
    return out


def seg_list_to_tensor(seg, has, h, w):
    """create a full size (all channels) segmentation mask from partial segmentation mask
    Args:
        seg: list of (h, w, <=n_cls), partial segmentation mask (might not have n_cls channels)
        has: (n, n_cls,) 
    
    Returns:
        seg: (n, n_cls, h, w) the full segmentation
    """
    b, n_cls = has.shape

    # get the device and dtype
    dtype = None
    device = None
    for each in seg:
        if each is not None:
            dtype = each.dtype
            device = each.device
            break

    seg_mask = torch.zeros(b, n_cls, h, w).type(dtype).to(device)
    for i in range(b):
        if seg[i] is not None:
            seg_mask[i, has[i]] = seg[i].permute([2, 0, 1])
    return seg_mask
