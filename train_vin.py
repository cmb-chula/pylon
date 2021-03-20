from trainer.start import *

from data.vin_data import *
from utils.callbacks import *
from utils.heatmap import *
from utils.localization import *
from utils.csv import *
from utils.exp_base import *


@dataclass
class VinConfig(Config):
    data_conf: VinDataConfig = VinDataConfig(split='v3')
    net_conf: PylonCustomConfig = None
    n_ep: int = 100
    n_eval_ep_cycle: int = 3
    lr: float = 1e-4
    rop_patience: int = 1
    rop_factor: float = 0.2
    log_to_file: bool = False


class VinExperiment(Experiment):
    def __init__(self, conf: VinConfig) -> None:
        super().__init__(conf, VinCombinedDataset, Trainer)
        self.conf = conf

    def make_dataset(self):
        self.dataset = VinCombinedDataset(self.conf.data_conf)
        self.train_loader = self.make_loader(self.dataset.train_data,
                                             shuffle=True,
                                             collate_fn=bbox_collate_fn)
        self.val_loader = self.make_loader(self.dataset.val_data,
                                           shuffle=False,
                                           collate_fn=bbox_collate_fn)
        self.test_loader = self.make_loader(self.dataset.test_data,
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
                    LocalizationAccCb(
                        keys=('pred_seg', 'bboxes'),
                        cls_id_to_name=cls_id_to_name,
                        conf=LocalizationAccConfig(intersect_thresholds=[]),
                    ),
                ],
            ),
        ]

    def warm_dataset(self):
        train_warm_loader = self.make_loader(self.dataset.train_data,
                                             shuffle=False,
                                             collate_fn=bbox_collate_fn)
        for each in tqdm(train_warm_loader):
            pass
        for each in tqdm(self.val_loader):
            pass
        for each in tqdm(self.test_loader):
            pass


def vin_256_baseline():
    out = []
    for seed in range(5):
        out.append(
            VinConfig(
                seed=seed,
                net_conf=BaselineModelConfig(n_out=15),
            ))
    return out


def vin_256_baseline_pretrain():
    out = []
    for seed in range(5):
        out.append(
            VinConfig(
                seed=seed,
                net_conf=BaselineModelConfig(
                    n_out=15,
                    pretrain_conf=PretrainConfig(
                        pretrain_name='baseline,nih14,256',
                        prefix='net.',
                    ),
                ),
            ))
    return out


def vin_256_li2018():
    out = []
    for seed in range(5):
        out.append(VinConfig(
            seed=seed,
            net_conf=Li2018Config(n_out=15),
        ))
    return out


def vin_256_fpn():
    out = []
    for seed in range(5):
        out.append(
            VinConfig(
                seed=seed,
                net_conf=FPNConfig(n_out=15,
                                   segment_block='custom',
                                   use_norm='batchnorm'),
            ))
    return out


def vin_256_pylon():
    out = []
    for seed in range(5):
        out.append(
            VinConfig(
                seed=seed,
                net_conf=PylonConfig(n_out=14, up_type='2layer'),
            ))
    return out


def vin_256_pretrain():
    out = []

    # normal transfer from pylon
    for seed in range(5):
        out.append(
            VinConfig(
                seed=seed,
                net_conf=PylonConfig(
                    n_out=15,
                    pretrain_conf=PretrainConfig(
                        pretrain_name='pylon,nih,256'),
                ),
            ))

    # two-phase
    for seed in range(5):
        pre_conf = VinConfig(
            seed=seed,
            net_conf=PylonConfig(
                n_out=15,
                pretrain_conf=PretrainConfig(pretrain_name='pylon,nih,256'),
                freeze='enc',
            ),
        )
        out.append(
            VinConfig(
                seed=seed,
                net_conf=PylonConfig(
                    n_out=15,
                    pretrain_conf=PretrainConfig(
                        pretrain_name='twophase',
                        path=get_pretrain_path(pre_conf.name),
                    ),
                ),
                pre_conf=pre_conf,
            ))

    return out


def vin_512_baseline():
    out = []
    for seed in range(5):
        out.append(
            VinConfig(
                seed=seed,
                data_conf=VinDataConfig(split='v3',
                                        trans_conf=TransformConfig(size=512)),
                net_conf=BaselineModelConfig(n_out=15),
            ))
    return out


def vin_512_li2018():
    out = []
    for seed in range(5):
        out.append(
            VinConfig(
                seed=seed,
                data_conf=VinDataConfig(split='v3',
                                        trans_conf=TransformConfig(size=512)),
                net_conf=Li2018Config(n_out=15),
            ))
    return out


def vin_512_fpn():
    out = []
    for seed in range(5):
        out.append(
            VinConfig(
                seed=seed,
                data_conf=VinDataConfig(split='v3',
                                        trans_conf=TransformConfig(size=512)),
                net_conf=FPNConfig(n_out=15,
                                   segment_block='custom',
                                   use_norm='batchnorm'),
            ))
    return out


def vin_512_pylon():
    out = []
    for seed in range(5):
        out.append(
            VinConfig(
                seed=seed,
                data_conf=VinDataConfig(split='v3',
                                        trans_conf=TransformConfig(size=512)),
                net_conf=PylonConfig(n_out=15),
            ))
    return out


def run_exp(conf: VinConfig):
    # conf.debug = True
    # conf.do_save = False
    # conf.resume = False
    # conf.log_to_file = False
    with global_queue(n=ENV.global_lock or 1,
                      enable=not conf.debug,
                      namespace=ENV.namespace):
        with cuda_round_robin(enable=not conf.debug,
                              namespace=ENV.namespace) as conf.device:
            with redirect_to_file(enable=conf.log_to_file):
                print(conf.name)
                exp = VinExperiment(conf)
                # exp.warm_dataset()
                exp.train()
                exp.test_auc()
                exp.test_loc()
                if conf.seed == 0:
                    exp.generate_picked_heatmap()
                    exp.generate_all_heatmap()


if __name__ == "__main__":
    confs = []
    # confs += vin_256_baseline()
    # confs += vin_256_li2018()
    # confs += vin_256_pylon()
    # confs += vin_256_fpn()

    # confs += vin_256_baseline_pretrain()
    # confs += vin_256_pretrain()

    # confs += vin_512_baseline()
    # confs += vin_512_li2018()
    # confs += vin_512_pylon()
    # confs += vin_512_fpn()

    multiprocess_map(run_exp,
                     confs,
                     num_workers=len(confs),
                     progress=True,
                     debug=False)
