from trainer.start import *
from utils.exp_base import *
from train_nih import *


@dataclass
class VinConfig(Config):
    n_ep: int = 100
    data_conf: VinDataConfig = VinDataConfig(bs=64, split='v3')
    net_conf: UnionModelConfig = None
    n_eval_ep_cycle: int = 3
    pre_conf: 'VinConfig' = None

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
        return VinExperiment(self)


class VinExperiment(Experiment):
    def __init__(self, conf: VinConfig) -> None:
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
                    LocalizationAccCb(
                        keys=('pred_seg', 'bboxes'),
                        cls_id_to_name=cls_id_to_name,
                        conf=LocalizationAccConfig(intersect_thresholds=[]),
                    ),
                ],
            ),
        ]


def vin_baseline(seed, size=256, bs=64):
    return [
        VinConfig(
            seed=seed,
            data_conf=VinDataConfig(bs=bs,
                                    trans_conf=XRayTransformConfig(size=size)),
            net_conf=BaselineModelConfig(n_out=15),
        )
    ]


def vin_li2018(seed, size=256, bs=64):
    return [
        VinConfig(
            seed=seed,
            data_conf=VinDataConfig(bs=bs,
                                    trans_conf=XRayTransformConfig(size=size)),
            net_conf=Li2018Config(n_out=15),
        )
    ]


def vin_fpn(seed,
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
        VinConfig(
            seed=seed,
            data_conf=VinDataConfig(bs=bs,
                                    trans_conf=XRayTransformConfig(size=size)),
            net_conf=FPNConfig(n_out=15,
                               segment_block=segment_block,
                               use_norm=use_norm,
                               n_group=n_group),
        )
    ]


def vin_pylon(seed, size=256, bs=64, up_type='2layer', **kwargs):
    return [
        VinConfig(
            seed=seed,
            data_conf=VinDataConfig(bs=bs,
                                    trans_conf=XRayTransformConfig(size=size)),
            net_conf=PylonConfig(n_in=1, n_out=15, up_type=up_type, **kwargs),
        )
    ]


def vin_baseline_transfer(seed, size=256, bs=64):
    out = []
    pre_conf = nih_baseline(seed, size, bs)[0]
    out.append(
        VinConfig(
            seed=seed,
            data_conf=VinDataConfig(bs=bs,
                                    trans_conf=XRayTransformConfig(size=size)),
            net_conf=BaselineModelConfig(
                n_out=15,
                pretrain_conf=PretrainConfig(
                    pretrain_name='nih',
                    path=get_pretrain_path(pre_conf.name),
                ),
            ),
            pre_conf=pre_conf,
        ))
    return out


def vin_pylon_transfer(seed, size=256, bs=64, up_type='2layer', **kwargs):
    out = []
    pre_conf = nih_pylon(seed, size, bs, up_type, **kwargs)[0]
    out.append(
        VinConfig(
            seed=seed,
            data_conf=VinDataConfig(bs=bs,
                                    trans_conf=XRayTransformConfig(size=size)),
            net_conf=PylonConfig(
                n_in=1,
                n_out=15,
                up_type=up_type,
                **kwargs,
                pretrain_conf=PretrainConfig(
                    pretrain_name='nih',
                    path=get_pretrain_path(pre_conf.name),
                ),
            ),
            pre_conf=pre_conf,
        ))
    return out


def vin_pylon_transfer_two_phase(seed,
                                 size=256,
                                 bs=64,
                                 up_type='2layer',
                                 **kwargs):
    out = []
    # train on NIH
    pre_conf = nih_pylon(seed, size, bs, up_type, **kwargs)[0]
    # train only the decoder
    first_phase_conf = VinConfig(
        seed=seed,
        data_conf=VinDataConfig(bs=bs,
                                trans_conf=XRayTransformConfig(size=size)),
        net_conf=PylonConfig(
            n_in=1,
            n_out=15,
            up_type=up_type,
            **kwargs,
            pretrain_conf=PretrainConfig(
                pretrain_name='nih',
                path=get_pretrain_path(pre_conf.name),
            ),
            freeze='enc',
        ),
        pre_conf=pre_conf,
    )
    # train all
    out.append(
        VinConfig(
            seed=seed,
            data_conf=VinDataConfig(bs=bs,
                                    trans_conf=XRayTransformConfig(size=size)),
            net_conf=PylonConfig(
                n_in=1,
                n_out=15,
                up_type=up_type,
                **kwargs,
                pretrain_conf=PretrainConfig(
                    pretrain_name='nih,twophase',
                    path=get_pretrain_path(first_phase_conf.name),
                ),
            ),
            pre_conf=first_phase_conf,
        ))
    return out
