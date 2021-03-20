from torch import Tensor
import numpy as np
import cv2


def overlay_cam(img, cam, weight=0.5, img_max=255.):
    """
    Red is the most important region
    Args:
        img: numpy array (h, w) 
        cam: numpy array (h, w)
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


def corners(bbox, base_size=None, tgt_size=None):
    """make the corners given a rectangle"""
    x, y, w, h, cls_id = bbox
    corners = np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h], [x,
                                                                         y]])
    if base_size is None or tgt_size is None:
        pass
    else:
        corners = corners * tgt_size / base_size
    return corners
