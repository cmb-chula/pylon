from trainer.start import *
from utils.exp_base import *


@dataclass
class NIH14Config(Config):
    n_ep: int = 40
    data_conf: NIH14DataConfig = NIH14DataConfig(bs=64)
    net_conf: UnionModelConfig = None
    pre_conf: 'NIH14Config' = None

    @property
    def name(self):
        if self.save_dir is not None:
            return self.save_dir
        a = f'{self.data_conf.name}'
        b = f'{self.net_conf.name}'
        if self.optimizier == 'pylonadam':
            b += f'_pylonadam_lr({",".join(str(lr) for lr in self.lr)})'
        else:
            b += f'_lr{self.lr}'
        b += f'term{self.lr_term}rop{self.rop_patience}fac{self.rop_factor}'
        if self.fp16:
            b += f'_fp16'
        c = f'{self.seed}'
        return '/'.join([a, b, c])

    def make_experiment(self):
        return NIH14Experiment(self)


class NIH14Experiment(Experiment):
    def __init__(self, conf: NIH14Config) -> None:
        super().__init__(conf, Trainer)
        self.conf = conf

    def make_dataset(self):
        self.data = self.conf.data_conf.make_dataset()
        self.train_loader = ConvertLoader(
            self.data.make_loader(self.data.train, shuffle=True),
            device=self.conf.device,
        )
        self.val_loader = ConvertLoader(
            self.data.make_loader(self.data.val, shuffle=False),
            device=self.conf.device,
        )
        self.test_loader = ConvertLoader(
            self.data.make_loader(self.data.test, shuffle=False),
            device=self.conf.device,
        )
        self.bbox_loader = ConvertLoader(
            self.data.make_loader(self.data.test_bbox, shuffle=False),
            device=self.conf.device,
        )

    def make_callbacks(self, trainer: Trainer):
        cls_id_to_name = self.data.test.id_to_cls
        return super().make_callbacks(trainer) + [
            ValidateCb(
                self.val_loader,
                n_ep_cycle=self.conf.n_eval_ep_cycle,
                name='val',
                callbacks=[
                    AvgCb(trainer.metrics),
                    AUROCCb(
                        keys=('pred', 'classification'),
                        cls_id_to_name=cls_id_to_name,
                    ),
                ],
            ),
            ValidateCb(
                self.bbox_loader,
                n_ep_cycle=self.conf.n_eval_ep_cycle,
                name='test',
                callbacks=[
                    LocalizationAccCb(
                        keys=('pred_seg', 'bboxes'),
                        cls_id_to_name=cls_id_to_name,
                        conf=LocalizationAccConfig(intersect_thresholds=[]),
                    )
                ],
            ),
        ]

    def test_loc(self):
        cls_id_to_name = self.data.test.id_to_cls
        callbacks = [
            ProgressCb('test'),
            LocalizationAccCb(
                keys=('pred_seg', 'bboxes'),
                cls_id_to_name=cls_id_to_name,
                conf=LocalizationAccConfig(
                    intersect_thresholds=(0.1, 0.25, 0.5),
                    # mode='iou',
                    mode='iobb_or_iou',
                ),
            )
        ]

        trainer = self.load_trainer()
        predictor = ValidatePredictor(trainer, callbacks)
        out, extras = predictor.predict(self.bbox_loader)
        out.update(extras)
        print(out)

        path = f'eval_loc/{self.conf.name}.csv'
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        df = DataFrame([out])
        df.to_csv(path, index=False)
        # group the seeds, it will be correct with the last seed
        group_seeds(dirname)

    def generate_all_heatmap(self):
        dataset = self.data.test_bbox
        dataset_ref = NIH14CombinedDataset(
            NIH14DataConfig(trans_conf=None)).test_bbox

        target_dir = f'figs/all/{self.conf.name}'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        self.generate_heatmap(dataset, dataset_ref, target_dir)


def nih_baseline(seed, size=256, bs=64):
    return [
        NIH14Config(
            seed=seed,
            data_conf=NIH14DataConfig(
                bs=bs, trans_conf=XRayTransformConfig(size=size)),
            net_conf=BaselineModelConfig(n_out=14),
        )
    ]


def nih_li2018(seed, size=256, bs=64):
    return [
        NIH14Config(
            seed=seed,
            data_conf=NIH14DataConfig(
                bs=bs, trans_conf=XRayTransformConfig(size=size)),
            net_conf=Li2018Config(n_out=14),
        )
    ]


def nih_pylon(seed, size=256, bs=64, up_type='2layer', **kwargs):
    return [
        NIH14Config(
            seed=seed,
            data_conf=NIH14DataConfig(
                bs=bs, trans_conf=XRayTransformConfig(size=size)),
            net_conf=PylonConfig(n_in=1, n_out=14, up_type=up_type, **kwargs),
        )
    ]


def nih_pan(seed, size=256, bs=64, use_gap=True):
    return [
        NIH14Config(
            seed=seed,
            data_conf=NIH14DataConfig(
                bs=bs, trans_conf=XRayTransformConfig(size=size)),
            net_conf=PANConfig(n_out=14, use_gap=use_gap),
        )
    ]


def nih_unet(seed, size=256, bs=64, n_dec_ch=(256, 128, 64, 64, 64)):
    return [
        NIH14Config(
            seed=seed,
            data_conf=NIH14DataConfig(
                bs=bs, trans_conf=XRayTransformConfig(size=size)),
            net_conf=UnetConfig(n_out=14, n_dec_ch=n_dec_ch),
        )
    ]


def nih_fpn(seed,
            size=256,
            bs=64,
            segment_block='custom',
            use_norm='batchnorm',
            n_group=None):
    """
    Args:
        segment_block: 'original', 'custom'
        use_norm: 'batchnorm', 'groupnorm' (on with 'custom')
    """
    return [
        NIH14Config(
            seed=seed,
            data_conf=NIH14DataConfig(
                bs=bs, trans_conf=XRayTransformConfig(size=size)),
            net_conf=FPNConfig(n_out=14,
                               segment_block=segment_block,
                               use_norm=use_norm,
                               n_group=n_group),
        )
    ]


def nih_deeplabv3(seed, size=256, bs=64, aspp_mode=None):
    """
    Args:
        aspp_mode: None, 'nogap', 'original'
    """
    data_conf = NIH14DataConfig(bs=bs,
                                trans_conf=XRayTransformConfig(size=size))
    out = []
    if aspp_mode is None:
        out.append(
            NIH14Config(
                seed=seed,
                data_conf=data_conf,
                net_conf=Deeplabv3Config(n_out=14),
            ))
    else:
        out.append(
            NIH14Config(
                seed=seed,
                data_conf=data_conf,
                net_conf=Deeplabv3CustomConfig(n_out=14, aspp_mode=aspp_mode),
            ))
    return out
