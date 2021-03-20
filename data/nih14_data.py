import os
from collections import defaultdict
from dataclasses import dataclass

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from trainer.config_base import *

from .transform import *

cv2.setNumThreads(1)
here = os.path.dirname(__file__)

# chestxray's
MEAN = [0.4984]
SD = [0.2483]

ID_TO_CLS = [
    'Atelectasis',
    'Cardiomegaly',
    'Effusion',
    'Infiltration',
    'Mass',
    'Nodule',
    'Pneumonia',
    'Pneumothorax',
    'Consolidation',
    'Edema',
    'Emphysema',
    'Fibrosis',
    'Pleural_Thickening',
    'Hernia',
]
CLS_TO_ID = {name: i for i, name in enumerate(ID_TO_CLS)}

# the spelling are not exact!
BBOX_CLS_NAME_TO_ID = {
    'Atelectasis': 0,
    'Cardiomegaly': 1,
    'Effusion': 2,
    'Infiltrate': 3,
    'Mass': 4,
    'Nodule': 5,
    'Pneumonia': 6,
    'Pneumothorax': 7,
}


@dataclass
class NIH14DataConfig(BaseConfig):
    trans_conf: TransformConfig = TransformConfig(mean=MEAN, std=SD)

    @property
    def name(self):
        return f'nih14_{self.trans_conf.name}'


class NIH14CombinedDataset:
    """Using the original test split, the train is around 70%, the val is <10%
    """
    def __init__(self, conf: NIH14DataConfig):
        self.conf = conf
        images_path = f'{here}/nih14/images'
        if conf.trans_conf is not None:
            train_transform = make_transform('train', conf.trans_conf)
            eval_transform = make_transform('eval', conf.trans_conf)
        else:
            train_transform = None
            eval_transform = None
        self.train_data = NIH14Dataset(
            f'{here}/nih14/original_split_train.csv', images_path,
            train_transform)
        self.val_data = NIH14Dataset(f'{here}/nih14/original_split_val.csv',
                                     images_path, eval_transform)
        self.test_data = NIH14Dataset(f'{here}/nih14/original_split_test.csv',
                                      images_path, eval_transform)
        self.test_bbox = NIH14Dataset(
            f'{here}/nih14/bbox_only.csv',
            images_path,
            eval_transform,
            bbox_csv=f'{here}/nih14/BBox_List_2017.csv')
        self.picked_data = NIH14Dataset(
            f'{here}/nih14/picked.csv',
            images_path,
            eval_transform,
            bbox_csv=f'{here}/nih14/BBox_List_2017.csv')


def make_coco_bbox_from_df(df, cls_name_map):
    bboxes = defaultdict(list)
    for i, row in df.iterrows():
        img_name, cls_name, x, y, w, h, *_ = row
        bboxes[img_name].append([x, y, w, h, cls_name_map[cls_name]])
    return bboxes


class NIH14Dataset(Dataset):
    def __init__(self, csv, img_dir, transform=None, bbox_csv=None):
        df = pd.read_csv(csv)
        self.df = df
        if bbox_csv is not None:
            bbox_df = pd.read_csv(bbox_csv)
            self.bbox = make_coco_bbox_from_df(bbox_df, BBOX_CLS_NAME_TO_ID)
        else:
            self.bbox = None
        self.img_dir = img_dir
        self.transform = transform

        self.cls_to_id = CLS_TO_ID
        self.id_to_cls = ID_TO_CLS

        support = []
        for i in range(len(self.id_to_cls)):
            support.append(df[self.id_to_cls[i]].sum())
        self.support = support

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        data = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, data['Image Index'])
        img = cv2_loader(img_path)
        if self.bbox is not None:
            bboxes = self.bbox[data['Image Index']]
        else:
            bboxes = []
        if self.transform:
            out = self.transform(image=img, bboxes=bboxes)
            img = out['image']
            bboxes = out['bboxes']

        labels = []
        for cls_name in self.id_to_cls:
            labels.append(data[cls_name])

        return {
            'img': img,
            'classification': torch.ByteTensor(labels),
            'bboxes': bboxes,
        }


def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f'cannot read {path}'
    return img


def bbox_collate_fn(data):
    out = {'img': [], 'classification': [], 'bboxes': []}
    for each in data:
        out['img'].append(each['img'])
        out['classification'].append(each['classification'])
        out['bboxes'].append(each['bboxes'])
    out['img'] = torch.stack(out['img'])
    out['classification'] = torch.stack(out['classification'])
    return out
