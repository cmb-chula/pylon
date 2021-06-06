import math

import cv2
from albumentations import *
from albumentations.pytorch import ToTensorV2
from trainer.start import *

cv2.setNumThreads(1)

# chestxray's
MEAN = (0.4984, )
SD = (0.2483, )


@dataclass
class XRayTransformConfig(BaseConfig):
    size: int = 256
    rotate: int = 90
    p_rotate: float = 0.5
    brightness: float = 0.5
    contrast: float = 0.5
    min_size: float = 0.7
    ratio_min: float = 3 / 4
    ratio_max: float = 4 / 3
    interpolation: str = 'cubic'
    bbox_min_visibility: float = 0.5
    mean: Tuple[float] = MEAN
    std: Tuple[float] = SD

    @property
    def name(self):
        a = []
        tmp = f'{self.size}min{self.min_size}'
        if not (math.isclose(self.ratio_min, 3 / 4)
                and math.isclose(self.ratio_max, 4 / 3)):
            tmp += f'rat({self.ratio_min:.2f},{self.ratio_max:.2f})'
        a.append(tmp)
        if self.p_rotate > 0:
            a.append(f'rot{self.rotate}p{self.p_rotate}')
        a.append(f'bc({self.brightness},{self.contrast})')
        a.append(f'{self.interpolation}')
        if self.bbox_min_visibility != 0.5:
            a.append(f'bboxmin{self.bbox_min_visibility}')
        return '-'.join(a)

    def make_transform(self, mode: str):
        """
        Args:
            mode: 'train' or 'eval'
        """
        inter_opts = {
            'linear': cv2.INTER_LINEAR,
            'cubic': cv2.INTER_CUBIC,
        }
        inter = inter_opts[self.interpolation]

        trans = []
        if mode == 'train':
            if self.rotate > 0:
                trans += [
                    Rotate(self.rotate,
                           border_mode=0,
                           p=self.p_rotate,
                           interpolation=inter)
                ]
            if self.min_size == 1:
                trans += [Resize(self.size, self.size, interpolation=inter)]
            else:
                trans += [
                    RandomResizedCrop(self.size,
                                      self.size,
                                      scale=(self.min_size, 1.0),
                                      ratio=(self.ratio_min, self.ratio_max),
                                      p=1.0,
                                      interpolation=inter)
                ]
            trans += [HorizontalFlip(p=0.5)]
            if self.contrast > 0 or self.brightness > 0:
                trans += [
                    RandomBrightnessContrast(self.brightness,
                                             self.contrast,
                                             p=0.5)
                ]
            trans += [Normalize(self.mean, self.std)]
        elif mode == 'eval':
            trans += [
                Resize(self.size, self.size, interpolation=inter),
                Normalize(self.mean, self.std),
            ]
        else:
            raise NotImplementedError()

        trans += [GrayToTensor()]
        return Compose(trans,
                       bbox_params=BboxParams(
                           format='coco',
                           min_visibility=self.bbox_min_visibility,
                       ))


class GrayToTensor(ToTensorV2):
    def apply(self, img, **params):
        # because of gray scale
        # we add an additional channel
        return torch.from_numpy(img).unsqueeze(0)
