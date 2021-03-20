import cv2
from albumentations import *
from albumentations.pytorch import ToTensorV2
from trainer.start import *

cv2.setNumThreads(1)

# chestxray's
MEAN = (0.4984, )
SD = (0.2483, )


@dataclass
class TransformConfig(BaseConfig):
    size: int = 256
    rotate: int = 90
    p_rotate: float = 0.5
    brightness: float = 0.5
    contrast: float = 0.5
    min_size: float = 0.7
    interpolation: str = 'cubic'
    bbox_min_visibility: float = 0.5
    mean: Tuple[float] = MEAN
    std: Tuple[float] = SD

    @property
    def name(self):
        a = []
        a.append(f'{self.size}min{self.min_size}')
        if self.p_rotate > 0:
            a.append(f'rot{self.rotate}p{self.p_rotate}')
        a.append(f'bc({self.brightness},{self.contrast})')
        a.append(f'{self.interpolation}')
        if self.bbox_min_visibility != 0.5:
            a.append(f'bboxmin{self.bbox_min_visibility}')
        return '-'.join(a)


def make_transform(augment, conf: TransformConfig):
    inter_opts = {
        'linear': cv2.INTER_LINEAR,
        'cubic': cv2.INTER_CUBIC,
    }
    inter = inter_opts[conf.interpolation]

    trans = []
    if augment == 'train':
        if conf.rotate > 0:
            trans += [
                Rotate(conf.rotate,
                       border_mode=0,
                       p=conf.p_rotate,
                       interpolation=inter)
            ]
        if conf.min_size == 1:
            trans += [Resize(conf.size, conf.size, interpolation=inter)]
        else:
            trans += [
                RandomResizedCrop(conf.size,
                                  conf.size,
                                  scale=(conf.min_size, 1.0),
                                  p=1.0,
                                  interpolation=inter)
            ]
        trans += [HorizontalFlip(p=0.5)]
        if conf.contrast > 0 or conf.brightness > 0:
            trans += [
                RandomBrightnessContrast(conf.brightness, conf.contrast, p=0.5)
            ]
        trans += [Normalize(conf.mean, conf.std)]
    elif augment == 'eval':
        trans += [
            Resize(conf.size, conf.size, interpolation=inter),
            Normalize(conf.mean, conf.std),
        ]
    else:
        raise NotImplementedError()

    trans += [GrayToTensor()]
    return Compose(trans,
                   bbox_params=BboxParams(
                       format='coco',
                       min_visibility=conf.bbox_min_visibility,
                   ))


class GrayToTensor(ToTensorV2):
    def apply(self, img, **params):
        # because of gray scale
        # we add an additional channel
        return torch.from_numpy(img).unsqueeze(0)
