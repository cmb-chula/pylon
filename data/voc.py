import cv2
from albumentations import *
from albumentations.pytorch import ToTensorV2
from torchvision.datasets import VOCDetection
from trainer.start import *

from data.common import BaseCombinedDataset

cv2.setNumThreads(1)
here = os.path.dirname(__file__)

ID_TO_CLS = [
    'background',
    'aeroplane',
    'bicycle',
    'bird',
    'boat',
    'bottle',
    'bus',
    'car',
    'cat',
    'chair',
    'cow',
    'diningtable',
    'dog',
    'horse',
    'motorbike',
    'person',
    'pottedplant',
    'sheep',
    'sofa',
    'train',
    'tvmonitor',
]
CLS_TO_ID = {name: i for i, name in enumerate(ID_TO_CLS)}


@dataclass
class VOCTransformConfig(BaseConfig):
    size: int = 256
    min_size: float = 0.5
    p_crop: float = 0.5
    p_flip: float = 0.5
    p_rotate: float = 0.5
    rotate: int = 10
    p_color: float = 0.5
    brightness: float = 0.2
    contrast: float = 0.2
    saturation: float = 0.2
    hue: float = 0.2
    interpolation: str = 'cubic'
    mean: Tuple[float] = (0.485, 0.456, 0.406)
    std: Tuple[float] = (0.229, 0.224, 0.225)

    @property
    def name(self):
        name = f'{self.size}'
        if self.p_crop > 0:
            name += f'min{self.min_size}'
        if self.p_flip > 0:
            name += f'flip{self.p_flip}'
        if self.p_rotate > 0:
            name += f'rot{self.rotate}p{self.p_rotate}'
        if self.p_color > 0:
            name += f'color({self.brightness},{self.contrast},{self.saturation},{self.hue})p{self.p_color}'
        return name

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
            if self.p_rotate > 0:
                trans.append(
                    Rotate(self.rotate,
                           border_mode=cv2.BORDER_CONSTANT,
                           value=0,
                           interpolation=inter))
            if self.p_crop > 0:
                trans.append(
                    Resize(int(self.size / self.min_size),
                           int(self.size / self.min_size),
                           interpolation=inter))
                trans.append(
                    RandomResizedCrop(self.size,
                                      self.size,
                                      scale=(self.min_size, 1),
                                      ratio=(1, 1),
                                      p=self.p_crop,
                                      interpolation=inter))
            trans.append(Resize(self.size, self.size, interpolation=inter))
            if self.p_flip > 0:
                trans.append(HorizontalFlip(p=self.p_flip))
            if self.p_color > 0:
                trans.append(
                    ColorJitter(self.brightness, self.contrast,
                                self.saturation, self.hue))
        elif mode == 'eval':
            trans.append(Resize(self.size, self.size, interpolation=inter))
        else:
            raise NotImplementedError()
        trans.append(Normalize(self.mean, self.std))
        trans.append(ToTensorV2())
        # we will use coco
        # need to make sure that it was converted
        # coco = (x, y, w, h)
        return Compose(trans, bbox_params=BboxParams(format='coco'))


@dataclass
class VOCDataConfig(BaseConfig):
    bs: int = 1
    root: str = f'{here}/voc2012'
    year: str = '2012'
    n_worker: int = ENV.num_workers
    trans_conf: VOCTransformConfig = VOCTransformConfig()

    @property
    def name(self):
        return f'bs{self.bs}_voc2012_{self.trans_conf.name}'

    def make_dataset(self):
        return VOCCombinedDataset(self)


class VOCCombinedDataset(BaseCombinedDataset):
    def __init__(self, conf: VOCDataConfig) -> None:
        self.conf = conf
        if conf.trans_conf is not None:
            train_trans = conf.trans_conf.make_transform('train')
            eval_trans = conf.trans_conf.make_transform('eval')
        else:
            train_trans = None
            eval_trans = None

        self.train = VOCDetectionDataset(conf, 'train', train_trans)
        self.val = VOCDetectionDataset(conf, 'val', eval_trans)


class VOCDetectionDataset(Dataset):
    def __init__(self, conf: VOCDataConfig, image_set: str, transform=None):
        super().__init__()
        self.conf = conf
        self.data = VOCDetection(root=conf.root,
                                 year=conf.year,
                                 image_set=image_set)

        self.cls_to_id = CLS_TO_ID
        self.id_to_cls = ID_TO_CLS

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img, attr = self.data[idx]
        img = np.array(img)

        bboxes = []
        for obj in attr['annotation']['object']:
            box = obj['bndbox']
            x, y, xx, yy = float(box['xmin']), float(box['ymin']), float(
                box['xmax']), float(box['ymax'])
            w, h = xx - x, yy - y
            cls_id = self.cls_to_id[obj['name']]
            bboxes.append((x, y, w, h, cls_id))

        if self.transform is not None:
            res = self.transform(image=img, bboxes=bboxes)
            img = res['image']
            bboxes = res['bboxes']

        labels = torch.LongTensor([0] * len(self.id_to_cls))
        for _, _, _, _, cls_id in bboxes:
            labels[cls_id] = 1

        return {
            'img': img,
            'bboxes': bboxes,
            'classification': labels,
            'attr': attr,
            'index': idx,
        }
