from trainer.start import *
from .localization import *


@dataclass
class LocalizationAccConfig(BaseConfig):
    prediction_threshold: float = 0.5
    intersect_thresholds: Tuple[float] = (0.1, )
    iobb_or_iou: bool = True
    apply_sigmoid: bool = True
    align_corners: bool = False


class LocalizationAccCb(CollectCb):
    def __init__(
            self,
            keys,
            cls_ids=None,
            cls_id_to_name=None,
            conf: LocalizationAccConfig = LocalizationAccConfig(),
    ):
        super().__init__(keys=keys)
        self.cls = cls_ids
        self.cls_id_to_name = cls_id_to_name
        self.keys = keys
        self.conf = conf
        self.w, self.h = None, None

    def on_forward_end(self, forward, buffer, i_itr, n_ep_itr, **kwargs):
        super().on_forward_end(forward, buffer, i_itr, n_ep_itr, **kwargs)
        n, c, h, w = forward['img'].shape
        # get the input size
        if self.w is None or self.h is None:
            self.w = w
            self.h = h

    def on_ep_end(self, buffer, i_itr, **kwargs):
        seg, bboxes = buffer[self.keys[0]], buffer[self.keys[1]]
        if self.conf.apply_sigmoid:
            seg = torch.sigmoid(seg)
        # we defer the upsampling to the point_loc_acc
        self.point_loc_acc_report(i_itr, seg, bboxes)
        for thresh in self.conf.intersect_thresholds:
            self.iobb_acc_report(i_itr, seg, bboxes, thresh)
        self._flush()

    def point_loc_acc_report(self, i_itr, seg, bboxes):
        scores_by_cls_id = point_loc_acc(
            seg,
            bboxes,
            conf=PointLocAccConfig(h=self.h,
                                   w=self.w,
                                   upsampling='bilinear',
                                   align_corners=self.conf.align_corners),
        )
        scores = np.array([each.mean() for each in scores_by_cls_id])
        weights = np.array([len(each) for each in scores_by_cls_id])
        weights = weights / weights.sum()

        if self.cls is None:
            # take classes with non-zero weights
            all_cls_id = []
            for i in range(seg.shape[1]):
                if weights[i] > 0:
                    all_cls_id.append(i)
        else:
            all_cls_id = self.cls

        weighted = (scores[all_cls_id] * weights[all_cls_id]).sum()
        macro = scores[all_cls_id].mean()

        bar = {
            'i_itr': i_itr,
            'point_acc_weighted': weighted,
            'point_acc_macro': macro,
        }
        self.add_to_bar_and_hist(bar)

        if self.cls_id_to_name is None:
            info = {f'point_acc_{i}': scores[i] for i in all_cls_id}
        else:
            info = {
                f'point_acc_{self.cls_id_to_name[i]}': scores[i]
                for i in all_cls_id
            }
        info['i_itr'] = i_itr
        self.add_to_hist(info)

    def iobb_acc_report(self, i_itr, seg, bboxes, intersect_threshold):
        scores_by_cls_id = iobb_acc(
            seg,
            bboxes,
            conf=IoBBConfig(
                prediction_threshold=self.conf.prediction_threshold,
                intersect_threshold=intersect_threshold,
                h=self.h,
                w=self.w,
                upsampling='bilinear',
                align_corners=self.conf.align_corners,
                iobb_or_iou=self.conf.iobb_or_iou,
            ),
        )
        scores = np.array([each.mean() for each in scores_by_cls_id])
        weights = np.array([len(each) for each in scores_by_cls_id])
        weights = weights / weights.sum()

        if self.cls is None:
            # take classes with non-zero weights
            all_cls_id = []
            for i in range(seg.shape[1]):
                if weights[i] > 0:
                    all_cls_id.append(i)
        else:
            all_cls_id = self.cls

        weighted = (scores[all_cls_id] * weights[all_cls_id]).sum()
        macro = scores[all_cls_id].mean()

        bar = {
            'i_itr': i_itr,
            f'iobb{intersect_threshold}_weighted': weighted,
            f'iobb{intersect_threshold}_macro': macro,
        }
        self.add_to_bar_and_hist(bar)

        if self.cls_id_to_name is None:
            info = {
                f'iobb{intersect_threshold}_{i}': scores[i]
                for i in all_cls_id
            }
        else:
            info = {
                f'iobb{intersect_threshold}_{self.cls_id_to_name[i]}':
                scores[i]
                for i in all_cls_id
            }
        info['i_itr'] = i_itr
        self.add_to_hist(info)
