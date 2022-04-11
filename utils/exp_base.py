import math
import multiprocessing as mp

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torch.optim as optim
from trainer.start import *

from data.nih14 import *
from data.vin import *
from data.voc import *
from model.baseline import *
from model.fpn import *
from pylon import *
from model.unet import *
from model.li2018 import *
from model.deeplab import *
from model.pan import *
from model.deeplab_custom import *
from utils.callbacks import *
from utils.csv import *
from utils.heatmap import *
from utils.localization import *

UnionModelConfig = Union[PylonConfig, BaselineModelConfig,
                         UnetConfig, FPNConfig, Li2018Config, PANConfig,
                         Deeplabv3Config]

UnionDataConfig = Union[NIH14DataConfig, VinDataConfig, VOCDataConfig]


@dataclass
class Config(BaseTrainerConfig):
    n_ep: int = None
    device: str = 'cuda:0'
    seed: int = 0
    data_conf: UnionDataConfig = None
    net_conf: UnionDataConfig = None
    optimizier: str = 'adam'
    lr: Union[float, Tuple[float]] = 1e-4
    lr_term: float = 1e-6
    rop_patience: int = 1
    rop_factor: float = 0.2
    n_worker: int = ENV.num_workers
    fp16: bool = True
    n_eval_ep_cycle: int = 1
    best_dir: str = 'best'
    save_dir: str = None
    resume: bool = True
    do_save: bool = True
    debug: bool = False
    pre_conf: 'Config' = None
    log_to_file: bool = False

    @property
    def name(self):
        raise NotImplementedError()

    def make_experiment(self):
        raise NotImplementedError()


class Trainer(BaseTrainer):
    def __init__(self, conf: Config, data: BaseCombinedDataset):
        super().__init__(conf)
        self.conf = conf
        self.data = data

    @property
    def metrics(self):
        return ['loss', 'loss_bbox', 'loss_pred']

    def forward_pass(self, data, **kwargs):
        res = self.net(**data)
        return {
            **data,
            **res.__dict__,
            'n': len(data['img']),
        }

    def make_default_callbacks(self):
        return super().make_default_callbacks() + [MovingAvgCb(self.metrics)]

    def make_net(self):
        return self.conf.net_conf.make_model()

    def make_opt(self, net):
        if self.conf.optimizier == 'pylonadam':
            return pylon_adam_optimizer(net, lrs=self.conf.lr)
        else:
            return optim.Adam(net.parameters(), lr=self.conf.lr)


class Experiment:
    def __init__(self, conf: Config, trainer_cls) -> None:
        self.conf = conf
        self.make_dataset()
        self.trainer_cls = trainer_cls
        if conf.fp16:
            self.trainer_cls = amp_trainer_mask(self.trainer_cls)

    def make_dataset(self):
        raise NotImplementedError()

    def train(self):
        if self.conf.pre_conf is not None:
            # two step training
            # create new experiment
            pre_exp = self.__class__(self.conf.pre_conf)
            pre_exp.train()

        print('running:', self.conf.name)
        set_seed(self.conf.seed)
        trainer = self.trainer_cls(self.conf, self.data)
        callbacks = trainer.make_default_callbacks() + self.make_callbacks(
            trainer)
        trainer.train(self.train_loader,
                      n_max_ep=self.conf.n_ep,
                      callbacks=callbacks)

    def make_callbacks(self, trainer: Trainer):
        callbacks = []
        callbacks += [
            LRReducePlateauCb('val_loss',
                              mode='min',
                              n_ep_cycle=self.conf.n_eval_ep_cycle,
                              patience=self.conf.rop_patience,
                              factor=self.conf.rop_factor),
            TerminateLRCb(self.conf.lr_term),
        ]

        if self.conf.do_save:
            callbacks += [
                LiveDataframeCb(f'save/{self.conf.name}/stats.csv'),
                AutoResumeCb(f'save/{self.conf.name}/checkpoints',
                             n_ep_cycle=self.conf.n_eval_ep_cycle,
                             keep_best=True,
                             metric='val_loss',
                             metric_best='min',
                             resume=self.conf.resume),
            ]

        return callbacks

    def load_trainer(self):
        trainer = self.trainer_cls(self.conf, self.data)
        if self.conf.best_dir != 'none':
            trainer.load(
                f'save/{self.conf.name}/checkpoints/{self.conf.best_dir}')
        return trainer

    def test(self):
        self.test_auc()
        self.test_loc()

    def test_auc(self):
        cls_id_to_name = self.data.test.id_to_cls
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
        out, extras = predictor.predict(self.test_loader)
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
        cls_id_to_name = self.data.test.id_to_cls
        callbacks = [
            ProgressCb('test'),
            LocalizationAccCb(
                keys=('pred_seg', 'bboxes'),
                cls_id_to_name=cls_id_to_name,
            )
        ]

        trainer = self.load_trainer()
        predictor = ValidatePredictor(trainer, callbacks)
        out, extras = predictor.predict(self.test_loader)
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

    def generate_picked_heatmap(self):
        dataset = self.data.picked
        conf_wo_augment = self.conf.data_conf.clone()
        conf_wo_augment.trans_conf = None
        dataset_ref = conf_wo_augment.make_dataset().picked

        target_dir = f'figs/picked/{self.conf.name}'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        self.generate_heatmap(dataset, dataset_ref, target_dir)

    def generate_all_heatmap(self):
        dataset = self.data.test
        conf_wo_augment = self.conf.data_conf.clone()
        conf_wo_augment.trans_conf = None
        dataset_ref = conf_wo_augment.make_dataset().test

        target_dir = f'figs/all/{self.conf.name}'
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        self.generate_heatmap(dataset, dataset_ref, target_dir)

    def generate_heatmap(self,
                         dataset: Dataset,
                         dataset_ref: Dataset,
                         target_dir: str,
                         plot_size: int = 7,
                         imgs_per_page: int = 10):
        """
        Args:
            dataset: may resize to fit the network
            dataset_ref: original resolution without augmentation
        """
        assert len(dataset) == len(dataset_ref)
        loader = ConvertLoader(self.data.make_loader(dataset, shuffle=False),
                               device=self.conf.device)
        trainer = self.load_trainer()
        # predict the segments
        predictor = BasePredictor(
            trainer,
            callbacks=[ProgressCb(desc='predict')],
            collect_keys=['img', 'pred', 'pred_seg', 'bboxes'],
        )
        out = predictor.predict(loader)

        pred = torch.sigmoid(out['pred'])
        pred_seg = torch.sigmoid(out['pred_seg'])
        # total plots = total img, cls of bounding boxes
        N = 0
        for i in range(pred.shape[0]):
            for cls_id in range(pred.shape[1]):
                bboxes = bbox_filter_by_cls_id(out['bboxes'][i], cls_id)
                if len(bboxes) > 0:
                    N += 1

        def create_ax(page_size, n_pages):
            # ax generator
            for page in range(n_pages):
                fig, ax = plt.subplots(nrows=page_size,
                                       ncols=2,
                                       figsize=(plot_size * 2,
                                                plot_size * page_size))
                for i in range(page_size):
                    # yield ax
                    yield ax[i]
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                fig.savefig(f'{target_dir}/{page}.jpg')
                plt.close(fig)

        # 10 imgs/page
        ax_itr = iter(create_ax(10, int(math.ceil(N / imgs_per_page))))
        # each plot = img with all bounding boxes of a single class
        # for each image
        scores = []
        for i in tqdm(range(pred.shape[0])):
            ref = dataset_ref[i]
            img = ref['img']
            # img = out['img'][i, 0]
            # img = img - img.min()
            # img = img / img.max()
            # img = img * 255.0
            img_bboxes = ref['bboxes']
            # img_bboxes = out['bboxes'][i]
            # assert len(img_bboxes) == len(ref['bboxes']), f'i: {i} {len(img_bboxes)} != {len(ref["bboxes"])}'
            h, w = img.shape
            for cls_id in range(pred.shape[1]):
                bboxes = bbox_filter_by_cls_id(img_bboxes, cls_id)
                if len(bboxes) == 0:
                    # no bounding box of this class
                    continue
                ax = next(ax_itr)
                ax[0].imshow(img, cmap='gray')
                # enlarge to the ref size
                # keeping the memory managable
                cam = F.interpolate(pred_seg[None, None, i, cls_id],
                                    size=(h, w),
                                    mode='bilinear',
                                    align_corners=False)[0, 0]
                row, col = argmax_2d_single(cam)
                cam = overlay_cam(img, cam)
                ax[1].imshow(cam)

                # bounding boxes
                correct = False
                for bbox in bboxes:
                    correct |= point_acc_single_bbox(row, col, bbox)
                    cn = corners(bbox)
                    ax[1].plot(cn[:, 0], cn[:, 1], color='white')
                scores.append(correct)

                ax[1].set_title(
                    f'{dataset.id_to_cls[cls_id]}: {pred[i, cls_id]:.2f} | point acc: {float(correct)}'
                )

        print('average score:', np.array(scores).mean())

        # drain the iterator
        while True:
            try:
                ax = next(ax_itr)
            except StopIteration:
                break

    def warm_dataset(self):
        train_warm_loader = self.data.make_loader(self.data.train,
                                                  shuffle=False)
        for each in tqdm(train_warm_loader):
            pass
        for each in tqdm(self.val_loader):
            pass
        for each in tqdm(self.test_loader):
            pass


class Run:
    def __init__(
        self,
        namespace: str = '',
        debug: bool = False,
        warm: bool = False,
        train: bool = True,
        test_auc: bool = True,
        test_loc: bool = False,
        gen_picked: bool = False,
        gen_all: bool = False,
    ) -> None:
        self.namespace = namespace
        self.debug = debug
        self.warm = warm
        self.train = train
        self.test_auc = test_auc
        self.test_loc = test_loc
        self.gen_picked = gen_picked
        self.gen_all = gen_all

    def __call__(self, conf: Config):
        if self.debug:
            conf.debug = True
            conf.do_save = False
            conf.resume = False
        else:
            conf.log_to_file = True
        with global_queue(enable=not conf.debug, namespace=self.namespace):
            with cuda_round_robin(enable=not conf.debug,
                                  namespace=self.namespace) as conf.device:
                with redirect_to_file(enable=conf.log_to_file):
                    print(conf.name)
                    exp = conf.make_experiment()
                    if self.warm:
                        exp.warm_dataset()
                    if self.train:
                        exp.train()
                    if self.test_auc:
                        exp.test_auc()
                    if self.test_loc:
                        exp.test_loc()
                    if conf.seed == 0:
                        if self.gen_picked:
                            exp.generate_picked_heatmap()
                        if self.gen_all:
                            exp.generate_all_heatmap()
