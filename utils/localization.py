def scale_to_one(pred):
    """scale so that the max = 1 in the prediction"""
    m, _ = pred.max(dim=-1, keepdim=True)
    m, _ = m.max(dim=-2, keepdim=True)
    pred = pred / m
    return pred


def ior(pred, seg):
    assert pred.shape == seg.shape
    insc = (pred * seg).sum(dim=[-2, -1])
    area = pred.sum(dim=[-2, -1])
    return insc / area


def iou(pred, seg):
    assert pred.shape == seg.shape
    insc = (pred * seg).sum(dim=[-2, -1])
    area = (pred.bool() | seg.bool()).sum(dim=[-2, -1])
    return insc / area


def ior_iou_acc(pred, seg, thresh):
    score = (ior(pred, seg) > thresh) | (iou(pred, seg) > thresh)
    return score.float()


def point_acc(pred, seg):
    m, _ = pred.max(dim=-1, keepdim=True)
    m, _ = m.max(dim=-2, keepdim=True)
    pred = (pred >= m).float()
    return ior(pred, seg)
