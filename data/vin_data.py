import os
from collections import defaultdict
from dataclasses import dataclass

import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
from trainer.config_base import *

from .nih14_data import cv2_loader

from .transform import *

cv2.setNumThreads(1)
here = os.path.dirname(__file__)

# chestxray's
MEAN = [0.4984]
SD = [0.2483]

ID_TO_CLS = [
    'Aortic enlargement',
    'Atelectasis',
    'Calcification',
    'Cardiomegaly',
    'Consolidation',
    'ILD',
    'Infiltration',
    'Lung Opacity',
    'Nodule/Mass',
    'Other lesion',
    'Pleural effusion',
    'Pleural thickening',
    'Pneumothorax',
    'Pulmonary fibrosis',
    'No finding',
]
CLS_TO_ID = {name: i for i, name in enumerate(ID_TO_CLS)}

SPLITS = {
    # train = 50%, val = 20%, test = 30%
    'v1': (
        'vin_cls_v1_train.csv',
        'vin_cls_v1_val.csv',
        'vin_cls_v1_test.csv',
        'v1_picked.csv',
    ),
    # train = 10%, val = 20%, test = 70%
    'v2': (
        'vin_cls_v2_train.csv',
        'vin_cls_v2_val.csv',
        'vin_cls_v2_test.csv',
        'v2_picked.csv',
    ),
    # train = 70%, val = 10%, test = 20%
    'v3': (
        'vin_cls_v3_train.csv',
        'vin_cls_v3_val.csv',
        'vin_cls_v3_test.csv',
        'v3_picked.csv',
    ),
}

IMAGES = {'1024ratio': ('images', 'images_ratio.csv')}


@dataclass
class VinDataConfig(BaseConfig):
    image: str = '1024ratio'
    file_ext: str = '.png'
    split: str = 'v3'
    trans_conf: TransformConfig = TransformConfig(mean=MEAN, std=SD)

    @property
    def name(self):
        return f'vin-{self.image}-{self.split}_{self.trans_conf.name}'


class VinCombinedDataset:
    """Using the original test split, the train is around 70%, the val is <10%
    """
    def __init__(self, conf: VinDataConfig):
        self.conf = conf
        image_dir, ratio_csv = IMAGES[conf.image]
        images_path = f'{here}/vin/{image_dir}'
        ratio_csv = f'{here}/vin/{ratio_csv}'

        if conf.trans_conf is not None:
            train_transform = make_transform('train', conf.trans_conf)
            eval_transform = make_transform('eval', conf.trans_conf)
        else:
            train_transform = None
            eval_transform = None

        train, val, test, picked = SPLITS[conf.split]
        bbox_csv = f'{here}/vin/train.csv'
        # preload and convert the bounding box
        bbox_df = pd.read_csv(bbox_csv)
        bbox = make_coco_bbox_from_df(bbox_df, CLS_TO_ID)
        self.train_data = VinDataset(f'{here}/vin/{train}',
                                     images_path,
                                     train_transform,
                                     bbox=bbox,
                                     ratio_csv=ratio_csv,
                                     file_ext=conf.file_ext)
        self.val_data = VinDataset(f'{here}/vin/{val}',
                                   images_path,
                                   eval_transform,
                                   bbox=bbox,
                                   ratio_csv=ratio_csv,
                                   file_ext=conf.file_ext)
        self.test_data = VinDataset(f'{here}/vin/{test}',
                                    images_path,
                                    eval_transform,
                                    bbox=bbox,
                                    ratio_csv=ratio_csv,
                                    file_ext=conf.file_ext)
        self.picked_data = VinDataset(f'{here}/vin/{picked}',
                                      images_path,
                                      eval_transform,
                                      bbox=bbox,
                                      ratio_csv=ratio_csv,
                                      file_ext=conf.file_ext)


def make_coco_bbox_from_df(df, cls_name_to_id):
    bboxes = defaultdict(list)
    for i, row in df.iterrows():
        img_name, cls_name, cls_id, rad_id, x, y, xx, yy, *_ = row
        if cls_name != 'No finding':
            w = xx - x
            h = yy - y
            bboxes[img_name].append([x, y, w, h, cls_name_to_id[cls_name]])
    return bboxes


class VinDataset(Dataset):
    def __init__(self,
                 csv,
                 img_dir,
                 transform=None,
                 bbox_csv=None,
                 bbox=None,
                 ratio_csv=None,
                 file_ext='.png'):
        df = pd.read_csv(csv)
        self.df = df

        if bbox is not None:
            # to speed up
            self.bbox = bbox
        elif bbox_csv is not None:
            bbox_df = pd.read_csv(bbox_csv)
            self.bbox = make_coco_bbox_from_df(bbox_df, CLS_TO_ID)
        else:
            self.bbox = None

        if ratio_csv is not None:
            ratio_df = pd.read_csv(ratio_csv)
            self.ratio = ratio_df
        else:
            self.ratio = None

        self.img_dir = img_dir
        self.transform = transform

        self.cls_to_id = CLS_TO_ID
        self.id_to_cls = ID_TO_CLS

        support = []
        for i in range(len(self.id_to_cls)):
            support.append(df[self.id_to_cls[i]].sum())
        self.support = support
        self.file_ext = file_ext

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        data = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, data['image_id'] + self.file_ext)
        img = cv2_loader(img_path)
        if self.bbox is not None:
            bboxes = self.bbox[data['image_id']]
            ratio = self.ratio[self.ratio['image_id'] ==
                               data['image_id']].iloc[0]['resize_ratio']
            bboxes = resize_bboxes(bboxes, ratio)
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


def resize_bboxes(bboxes, ratio):
    out = []
    for bbox in bboxes:
        x, y, w, h, cls_id = bbox
        box = (np.array([x, y, w, h]) * ratio).astype(int)
        out.append([*box, cls_id])
    return out