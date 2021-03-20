from typing import List, Tuple

import numpy as np
from torch import Tensor
import torch.nn.functional as F
from dataclasses import dataclass
import torch
from tqdm.autonotebook import tqdm

torch.set_num_threads(1)


def scale_to_one(seg):
    """scale so that the max = 1 in the prediction"""
    m, _ = seg.max(dim=-1, keepdim=True)
    m, _ = m.max(dim=-2, keepdim=True)
    seg = seg / m
    return seg


def _iobb(pred, gt):
    assert pred.shape == gt.shape
    insc = (pred * gt).sum(dim=[-2, -1])
    area = pred.sum(dim=[-2, -1])
    return insc / area


def _iobb_acc(pred, gt, thresh):
    score = _iobb(pred, gt) > thresh
    return score.float()


def _iou(pred, gt):
    assert pred.shape == gt.shape
    insc = (pred * gt).sum(dim=[-2, -1])
    area = (pred.bool() | gt.bool()).sum(dim=[-2, -1])
    return insc / area


def _iobb_iou_acc(pred, gt, thresh):
    score = (_iobb(pred, gt) > thresh) | (_iou(pred, gt) > thresh)
    return score.float()


def argmax_2d(x):
    n, c, h, w = x.shape
    argmax = x.view(n, c, h * w).argmax(dim=-1)
    row = argmax // w
    col = argmax % w
    return row, col


def argmax_2d_single(x) -> Tuple[int, int]:
    h, w = x.shape
    argmax = x.view(h * w).argmax(dim=-1)
    row = argmax // w
    col = argmax % w
    return row.item(), col.item()


@dataclass
class PointLocAccConfig:
    h: int
    w: int
    upsampling: str
    align_corners: bool


def point_acc_single_bbox(row, col, bbox: Tuple[float]):
    x, y, w, h, cls_id = bbox
    return x <= col <= x + w and y <= row <= y + h


def point_loc_acc_by_img(row, col, img_cls_bboxes: List[Tuple[float]]):
    # if the prediction is within any of the bounding boxes, it gets full mark
    for bbox in img_cls_bboxes:
        if point_acc_single_bbox(row, col, bbox):
            return 1.
    return 0.


def point_loc_acc_by_cls(seg: Tensor, cls_bboxes: List[List[Tuple[float]]],
                         conf: PointLocAccConfig):
    assert len(seg) == len(cls_bboxes)
    n, h, w = seg.shape
    out = []
    # by image
    for i in range(len(seg)):
        # skip images with no bbox (of this class)
        if len(cls_bboxes[i]) == 0:
            continue
        # lazy upsampling due to memory contraints
        single_seg = F.interpolate(seg[None, i:i + 1, :, :], (conf.h, conf.w),
                                   mode=conf.upsampling,
                                   align_corners=conf.align_corners)
        row, col = argmax_2d_single(single_seg[0, 0])
        out.append(point_loc_acc_by_img(
            row,
            col,
            cls_bboxes[i],
        ))
    out = np.array(out)
    return out


def bbox_filter_by_cls_id(img_bboxes: List[Tuple[float]], cls_id):
    """returns bboxes of the same class"""
    out = []
    for bbox in img_bboxes:
        x, y, w, h, bbox_cls_id = bbox
        if bbox_cls_id == cls_id:
            out.append(bbox)
    return out


def point_loc_acc(
        seg: Tensor,
        bboxes: List[List[Tuple[float]]],
        conf: PointLocAccConfig,
) -> List[np.ndarray]:
    """
    Args:
        seg: (n, c, h, w) doesn't have to match the bounding box size yet
        bboxes: List of image bboxes [n]
        conf: supplied upsampling information
    Returns: List of array of scores by class id
    """
    N, C, H, W = seg.shape

    # look by class
    acc_by_cls_id = [None for i in range(C)]  # by cls_id
    for cls_id in tqdm(range(C), desc='point_loc_acc'):
        cls_bboxes = [
            bbox_filter_by_cls_id(img_bboxes, cls_id) for img_bboxes in bboxes
        ]
        acc_by_cls_id[cls_id] = point_loc_acc_by_cls(seg[:, cls_id],
                                                     cls_bboxes, conf)
    return acc_by_cls_id


@dataclass
class IoBBConfig:
    prediction_threshold: float
    intersect_threshold: float
    h: int
    w: int
    upsampling: str
    align_corners: bool
    iobb_or_iou: bool


def iobb_acc_by_image(seg, img_cls_bboxes: List[Tuple[float]],
                      conf: IoBBConfig):
    """
    Args:
        seg: (h, w)
        img_cls_bboxes: list of bboxes of a particular image
    """
    canvas = torch.zeros(conf.h, conf.w)
    # if the prediction is within any of the bounding boxes, it gets full mark
    for bbox in img_cls_bboxes:
        x, y, w, h, *_ = bbox
        x, y, w, h = np.array([x, y, w, h]).astype(int)
        canvas[y:y + h, x:x + w] = 1.

    # intersection
    seg = scale_to_one(seg)
    seg = (seg > conf.prediction_threshold).float()
    if conf.iobb_or_iou:
        return _iobb_iou_acc(seg, canvas, conf.intersect_threshold)
    else:
        return _iobb_acc(seg, canvas, conf.intersect_threshold)


def iobb_acc_by_cls(seg: Tensor, cls_bboxes: List[List[Tuple[float]]],
                    conf: IoBBConfig):
    assert len(seg) == len(cls_bboxes)
    n, h, w = seg.shape
    out = []
    # by image
    for i in range(len(seg)):
        # skip images with no bbox (of this class)
        if len(cls_bboxes[i]) == 0:
            continue
        # lazy upsampling due to memory contraints
        single_seg = F.interpolate(seg[None, i:i + 1, :, :], (conf.h, conf.w),
                                   mode=conf.upsampling,
                                   align_corners=conf.align_corners)
        out.append(
            iobb_acc_by_image(
                single_seg[0, 0],
                cls_bboxes[i],
                conf=conf,
            ))
    out = np.array(out)
    return out


def iobb_acc(
        seg: Tensor,
        bboxes: List[List[Tuple[float]]],
        conf: IoBBConfig,
) -> List[np.ndarray]:
    """
    Args:
        seg: (n, c, h, w) doesn't have to match the bounding box size yet
        bboxes: List of image bboxes [n]
        conf: supplied upsampling information
    Returns: List of array of scores by class id
    """
    N, C, H, W = seg.shape

    # look by class
    acc_by_cls_id = [None for i in range(C)]  # by cls_id
    for cls_id in tqdm(range(C), desc='iobb_acc'):
        cls_bboxes = [
            bbox_filter_by_cls_id(img_bboxes, cls_id) for img_bboxes in bboxes
        ]
        acc_by_cls_id[cls_id] = iobb_acc_by_cls(seg[:, cls_id], cls_bboxes,
                                                conf)
    return acc_by_cls_id
