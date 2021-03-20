import torch
from trainer.start import *
from utils.exp_base import *
"""
Problem:
RuntimeError: received 0 items of ancdata
https://github.com/pytorch/pytorch/issues/973
"""
torch.multiprocessing.set_sharing_strategy('file_system')


@dataclass
class NIH14Config(Config):
    bs: int = 64
    n_ep: int = 40
    data_conf: NIH14DataConfig = NIH14DataConfig()
    net_conf: UnionModelConfig = None
    pre_conf: 'NIH14Config' = None
    log_to_file: bool = False


class NIH14Experiment(Experiment):
    def __init__(self, conf: NIH14Config) -> None:
        super().__init__(conf, NIH14CombinedDataset, Trainer)
        self.conf = conf

    def make_dataset(self):
        self.dataset = NIH14CombinedDataset(self.conf.data_conf)
        self.train_loader = self.make_loader(self.dataset.train_data,
                                             shuffle=True)
        self.val_loader = self.make_loader(self.dataset.val_data,
                                           shuffle=False)
        self.test_loader = self.make_loader(self.dataset.test_data,
                                            shuffle=False)
        self.bbox_loader = self.make_loader(self.dataset.test_bbox,
                                            shuffle=False,
                                            collate_fn=bbox_collate_fn)

    def make_callbacks(self):
        cls_id_to_name = self.dataset.test_data.id_to_cls
        return super().make_callbacks() + [
            ValidateCb(
                self.val_loader,
                n_ep_cycle=self.conf.n_eval_ep_cycle,
                name='val',
                callbacks=[
                    AvgCb('loss'),
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
        cls_id_to_name = self.dataset.test_data.id_to_cls
        callbacks = [
            ProgressCb('test'),
            LocalizationAccCb(
                keys=('pred_seg', 'bboxes'),
                cls_id_to_name=cls_id_to_name,
                conf=LocalizationAccConfig(intersect_thresholds=(0.1, 0.25,
                                                                 0.5)),
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
        dataset = self.dataset.test_bbox
        dataset_ref = NIH14CombinedDataset(
            NIH14DataConfig(trans_conf=None)).test_bbox

        target_dir = f'figs/all/{self.conf.name}'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        self.generate_heatmap(dataset, dataset_ref, target_dir)


def nih_256_baseline():
    out = []
    for seed in range(1):
        out.append(
            NIH14Config(
                seed=seed,
                net_conf=BaselineModelConfig(n_out=14),
            ))
    return out


def nih_256_li2018():
    out = []
    for seed in range(5):
        out.append(NIH14Config(
            seed=seed,
            net_conf=Li2018Config(n_out=14),
        ))
    return out


def nih_256_pylon():
    out = []
    for seed in range(5):
        out.append(NIH14Config(
            seed=seed,
            net_conf=PylonConfig(n_out=14),
        ))
    return out


def nih_256_pan():
    out = []
    for seed in range(5):
        out.append(NIH14Config(
            seed=seed,
            net_conf=PANConfig(n_out=14),
        ))

    return out


def nih_256_unet():
    out = []
    for seed in range(5):
        out.append(
            NIH14Config(
                seed=seed,
                net_conf=UnetConfig(n_out=14, n_dec_ch=(256, 128, 64, 64, 64)),
            ))
    return out


def nih_256_fpn():
    out = []
    for seed in range(5):
        out.append(
            NIH14Config(
                seed=seed,
                net_conf=FPNConfig(n_out=14,
                                   segment_block='custom',
                                   use_norm='batchnorm'),
            ))
    return out


def nih_256_deeplabv3():
    out = []
    for seed in range(5):
        out.append(NIH14Config(
            seed=seed,
            net_conf=Deeplabv3Config(n_out=14),
        ))
        out.append(
            NIH14Config(
                seed=seed,
                net_conf=Deeplabv3CustomConfig(n_out=14, aspp_mode='nogap'),
            ))
        # out.append(
        #     NIH14Config(
        #         seed=seed,
        #         net_conf=Deeplabv3CustomConfig(n_out=14, aspp_mode='original'),
        #     ))

    return out


def nih_512_baseline():
    out = []
    for seed in range(5):
        out.append(
            NIH14Config(
                seed=seed,
                data_conf=NIH14DataConfig(TransformConfig(size=512)),
                net_conf=BaselineModelConfig(n_out=14),
            ))
    return out


def nih_512_pylon():
    out = []
    for seed in range(5):
        out.append(
            NIH14Config(
                seed=seed,
                data_conf=NIH14DataConfig(TransformConfig(size=512)),
                net_conf=PylonConfig(n_out=14,
                                     backbone=backbone,
                                     up_type='1layer'),
            ))
        out.append(
            NIH14Config(
                seed=seed,
                data_conf=NIH14DataConfig(TransformConfig(size=512)),
                net_conf=PylonConfig(n_out=14,
                                     backbone=backbone,
                                     up_type='2layer'),
            ))
    return out


def nih_512_li2018():
    out = []
    for seed in range(5):
        out.append(
            NIH14Config(
                seed=seed,
                data_conf=NIH14DataConfig(TransformConfig(size=512)),
                net_conf=Li2018Config(n_out=14),
            ))
    return out


def nih_512_fpn():
    out = []
    for seed in range(5):
        out.append(
            NIH14Config(
                seed=seed,
                data_conf=NIH14DataConfig(TransformConfig(size=512)),
                net_conf=FPNConfig(n_out=14,
                                   segment_block='custom',
                                   use_norm='batchnorm'),
            ))
    return out


def run_exp(conf: NIH14Config):
    # conf.debug = True
    # conf.do_save = False
    # conf.resume = False
    # conf.log_to_file = True
    with global_queue(n=ENV.global_lock or 1,
                      enable=not conf.debug,
                      namespace=ENV.namespace):
        with cuda_round_robin(enable=not conf.debug,
                              namespace=ENV.namespace) as conf.device:
            with redirect_to_file(enable=conf.log_to_file):
                print(conf.name)
                exp = NIH14Experiment(conf)
                # exp.warm_dataset()
                exp.train()
                exp.test_auc()
                exp.test_loc()
                if conf.seed == 0:
                    exp.generate_picked_heatmap()
                    exp.generate_all_heatmap()


if __name__ == "__main__":
    confs = []
    # confs += nih_256_baseline()
    # confs += nih_256_li2018()
    # confs += nih_256_pylon()
    # confs += nih_256_pan()
    # confs += nih_256_unet()
    # confs += nih_256_fpn()
    # confs += nih_256_deeplabv3()

    # confs += nih_512_baseline()
    # confs += nih_512_li2018()
    # confs += nih_512_pylon()
    # confs += nih_512_fpn()

    multiprocess_map(run_exp,
                     confs,
                     num_workers=len(confs),
                     progress=True,
                     debug=False)
