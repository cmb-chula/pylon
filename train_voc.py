from trainer.start import *
from utils.exp_base import *


@dataclass
class VOCConfig(Config):
    n_ep: int = 40
    data_conf: VOCDataConfig = VOCDataConfig(bs=64)
    net_conf: UnionModelConfig = None
    pre_conf: 'VOCConfig' = None

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
        return VOCExperiment(self)


class VOCExperiment(Experiment):
    def __init__(self, conf: VOCConfig) -> None:
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

    def make_callbacks(self, trainer: Trainer):
        cls_id_to_name = self.data.val.id_to_cls
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
                    )
                ],
            ),
        ]

    def test_auc(self):
        cls_id_to_name = self.data.val.id_to_cls
        callbacks = [
            ProgressCb('test'),
            AvgCb('loss'),
            AUROCCb(
                keys=('pred', 'classification'),
                cls_id_to_name=cls_id_to_name,
            ),
        ]

        trainer = self.load_trainer()
        predictor = ValidatePredictor(trainer, callbacks)
        out, extras = predictor.predict(self.val_loader)
        out.update(extras)
        print(out)

        path = f'eval_auc/{self.conf.name}.csv'
        dirname = os.path.dirname(path)
        if not os.path.exists(dirname):
            os.makedirs(dirname)
        df = DataFrame([out])
        df.to_csv(path, index=False)
        # group the seeds, it will be correct with the last seed
        group_seeds(dirname)

    def test_loc(self):
        cls_id_to_name = self.data.val.id_to_cls
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
        out, extras = predictor.predict(self.val_loader)
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
        raise NotImplementedError()

    def generate_picked_heatmap(self):
        raise NotImplementedError()


def voc_baseline(seed, size=256, bs=64):
    return [
        VOCConfig(
            seed=seed,
            data_conf=VOCDataConfig(bs=bs,
                                    trans_conf=VOCTransformConfig(size=size)),
            net_conf=BaselineModelConfig(n_out=21, n_in=3),
        )
    ]


def voc_li2018(seed, size=256, bs=64):
    return [
        VOCConfig(
            seed=seed,
            data_conf=VOCDataConfig(bs=bs,
                                    trans_conf=VOCTransformConfig(size=size)),
            net_conf=Li2018Config(n_out=21, n_in=3),
        )
    ]


def voc_fpn(seed,
            size=256,
            bs=64,
            segment_block='custom',
            use_norm='batchnorm',
            n_group=None):
    return [
        VOCConfig(
            seed=seed,
            data_conf=VOCDataConfig(bs=bs,
                                    trans_conf=VOCTransformConfig(size=size)),
            net_conf=FPNConfig(n_out=21,
                               n_in=3,
                               segment_block=segment_block,
                               use_norm=use_norm,
                               n_group=n_group),
        )
    ]


def voc_pylon_two_phase(seed, size=256, bs=64, up_type='2layer', **kwargs):
    data_conf = VOCDataConfig(bs=bs, trans_conf=VOCTransformConfig(size=size))
    out = []
    # train only the decoder
    first_phase_config = VOCConfig(
        seed=seed,
        data_conf=data_conf,
        net_conf=PylonConfig(
            n_in=3,
            n_out=21,
            up_type=up_type,
            **kwargs,
            freeze='enc',
        ),
    )
    # train the whole network
    out.append(
        VOCConfig(
            seed=seed,
            data_conf=data_conf,
            net_conf=PylonConfig(
                n_in=3,
                n_out=21,
                up_type=up_type,
                **kwargs,
                pretrain_conf=PretrainConfig(
                    pretrain_name='twophase',
                    path=get_pretrain_path(first_phase_config.name),
                ),
            ),
            pre_conf=first_phase_config,
        ))

    return out
