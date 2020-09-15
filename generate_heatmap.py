import math
import os

import cv2
from matplotlib import image
import matplotlib.pyplot as plt
import torch.nn.functional as F
from tqdm import tqdm

from dataset_segment import *
from train import BinaryClassificationTrainer, make_net
from trainer.all import *
from utils.loader import *


def overlay_cam(img, cam, weight=0.5, img_max=255.):
    """
    Red is the most important region
    Args:
        img: numpy array (h, w) or (h, w, 3)
    """
    if isinstance(img, Tensor):
        img = img.cpu().numpy()
    if isinstance(cam, Tensor):
        cam = cam.detach().cpu().numpy()

    if len(img.shape) == 2:
        h, w = img.shape
        img = img.reshape(h, w, 1)
        img = np.repeat(img, 3, axis=2)

    h, w, c = img.shape

    # normalize the cam
    x = cam
    x = x - x.min()
    x = x / x.max()
    # resize the cam
    x = cv2.resize(x, (w, h))
    x = x - x.min()
    x = x / x.max()
    # coloring the cam
    x = cv2.applyColorMap(np.uint8(255 * (1 - x)), cv2.COLORMAP_JET)
    x = np.float32(x) / 255.

    # overlay
    x = img / img_max + weight * x
    x = x / x.max()
    return x


def enclose_rect(mask: np.ndarray):
    """creating a list of bounding rectangles"""
    mask = (1 - mask).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_LIST,
                                   cv2.CHAIN_APPROX_SIMPLE)
    out = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        out.append((x, y, w, h))
    return out


def corners(rect, base_size=256, tgt_size=1024):
    """make the corners given a rectangle"""
    x, y, w, h = rect
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x,
                                                                         y]])
    corners = corners * 1024 / 256
    return corners


picked_files = [
    '00000032_037.png',  # Infiltration
    '00000150_002.png',  # Pneumonia
    '00000211_041.png',  # Cardiomegaly
    '00000468_017.png',  # Atelectasis
    '00001075_024.png',  # Mass
    '00008547_001.png',  # Effusion
    '00004547_003.png',  # Nodule
    '00008522_032.png',  # Cardiomegaly
    '00002106_000.png',  # Mass, Pneumothorax
    '00004911_018.png',  # Nodule
    '00002350_001.png',  # Atelectasis
    '00003440_000.png',  # Mass, Atelectasis
    '00002533_002.png',  # Effusion
    '00006736_000.png',  # Nodule
    '00002856_009.png',  # Atelectasis
    '00002980_000.png',  # Pneumonia
    '00008386_000.png',  # Nodule
    '00008008_027.png',  # Mass
]


def generate(checkpoint,
             picked=True,
             images_path='data/images',
             size=256,
             interpolation='cubic',
             dev='cuda:0'):
    eval_transform = make_transform('eval',
                                    size=size,
                                    interpolation=interpolation)
    dataset_ori = ChestXRay14Segment(images_path, None)
    dataset = ChestXRay14Segment(images_path, eval_transform)
    cls_names = dataset.segment_only_data.i_to_l

    # only picked images
    if picked:
        dataset_ori.segment_only_data.names = picked_files
        dataset.segment_only_data.names = picked_files

    loader = ConvertLoader(
        DataLoader(dataset.segment_only_data,
                   batch_size=32,
                   collate_fn=collate_fn), dev)

    # load the checkpoint
    trainer = BinaryClassificationTrainer(make_net('resnet50'), None, dev)
    trainer.load(checkpoint)

    # predict the segments
    predictor = BasePredictor(
        trainer,
        callbacks=[ProgressCb(desc='predict')],
        collect_keys=['pred', 'pred_seg', 'seg', 'has', 'evidence'],
    )
    out = predictor.predict(loader)

    # collect data
    seg = seg_list_to_tensor(out['seg'], out['has'], w=size, h=size)
    pred = torch.sigmoid(out['pred'])
    pred_seg = torch.sigmoid(out['pred_seg'])
    pred_seg = F.interpolate(pred_seg,
                             size=size,
                             mode='bilinear',
                             align_corners=False)
    has = out['has']

    # total plots
    n = int(has.sum())

    def create_ax(page_size, n_pages):
        # ax generator
        for page in range(n_pages):
            fig, ax = plt.subplots(nrows=page_size,
                                   ncols=2,
                                   figsize=(10, 5 * page_size))
            for i in range(page_size):
                yield ax[i]
            if picked:
                dirname = 'figs/picked'
            else:
                dirname = 'figs/all'
            if not os.path.exists(dirname):
                os.makedirs(dirname)
            fig.savefig(f'{dirname}/{page}.jpg')
            plt.close(fig)

    # 10 imgs/page
    itr = iter(create_ax(10, int(math.ceil(n / 10))))
    for i in tqdm(range(has.shape[0]), desc='plotting'):
        for j in range(has.shape[1]):
            if has[i, j]:
                ax = next(itr)
                img = dataset_ori.segment_only_data[i]['img']
                ax[0].imshow(img, cmap='gray')
                cam = overlay_cam(img, pred_seg[i, j])
                ax[1].imshow(cam)

                # bounding boxes
                rects = enclose_rect(seg[i, j].numpy())
                for rect in rects[:-1]:
                    cn = corners(rect, base_size=size, tgt_size=1024)
                    ax[1].plot(cn[:, 0], cn[:, 1], color='white')

                ax[1].set_title(f'{cls_names[j]}: {pred[i, j]:.2f}')

    # drain the iterator
    while True:
        try:
            ax = next(itr)
        except StopIteration:
            break


if __name__ == "__main__":
    checkpoint = 'save/pylon/0/best'
    # picked images
    generate(checkpoint, picked=True)
    # all images
    generate(checkpoint, picked=False)
