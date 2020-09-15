import os

import cv2
import pandas as pd
import torch
from albumentations import *
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

cv2.setNumThreads(1)
here = os.path.dirname(__file__)

# chestxray's
MEAN = [0.4984]
SD = [0.2483]


class ChestXRay14OriginalSplit:
    """Using the original test split, the train is around 70%, the val is <10%
    """
    def __init__(self, images_path, train_transform, eval_transform):
        self.train_data = ChestXRay14CSV(
            f'{here}/data/original_split_train.csv', images_path,
            train_transform)
        self.val_data = ChestXRay14CSV(f'{here}/data/original_split_val.csv',
                                       images_path, eval_transform)
        self.test_data = ChestXRay14CSV(f'{here}/data/original_split_test.csv',
                                        images_path, eval_transform)


class ChestXRay14CSV(Dataset):
    def __init__(self, csv, img_dir, transform=None):
        df = pd.read_csv(csv)

        self.df = df
        self.img_dir = img_dir
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

        support = []
        for i in range(len(self.i_to_l)):
            support.append(df[self.i_to_l[i]].sum())
        self.support = support

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        data = self.df.iloc[i]
        img_path = os.path.join(self.img_dir, data['Image Index'])
        img = cv2_loader(img_path)
        assert img is not None, f'cannot read {img_path}'
        if self.transform:
            _res = self.transform(image=img)
            img = _res['image']

        labels = []
        for i, k in self.i_to_l.items():
            labels.append(data[k])

        return {
            'img': img,
            'evidence': torch.tensor(labels).float(),
        }


def make_transform(
        augment,
        size=256,
        rotate=90,
        p_rotate=0.5,
        brightness=0.5,
        contrast=0.5,
        min_size=0.7,
        interpolation='cubic',
):
    inter_opts = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
    }
    inter = inter_opts[interpolation]

    trans = []
    if augment == 'common':
        if rotate > 0:
            trans += [
                Rotate(rotate, border_mode=0, p=p_rotate, interpolation=inter)
            ]
        if min_size == 1:
            trans += [Resize(size, size, interpolation=inter)]
        else:
            trans += [
                RandomResizedCrop(size,
                                  size,
                                  scale=(min_size, 1.0),
                                  p=1.0,
                                  interpolation=inter)
            ]
        trans += [HorizontalFlip(p=0.5)]
        if contrast > 0 or brightness > 0:
            trans += [RandomBrightnessContrast(brightness, contrast, p=0.5)]
        trans += [Normalize(MEAN, SD)]
    elif augment == 'eval':
        trans += [
            Resize(size, size, interpolation=inter),
            Normalize(MEAN, SD),
        ]
    else:
        raise NotImplementedError()

    trans += [GrayToTensor()]
    return Compose(trans)


def cv2_loader(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    return img


class GrayToTensor(ToTensorV2):
    def apply(self, img, **params):
        # because of gray scale
        # we add an additional channel
        return torch.from_numpy(img).unsqueeze(0)
