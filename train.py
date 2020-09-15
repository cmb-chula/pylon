import multiprocessing as mp
from functools import partial

import torch
import torch.nn.functional as F
import torch.optim as optim

from dataset import *
from dataset_segment import *
from pylon import *
from trainer.all import *
from trainer.callbacks.all import *
from utils.loader import *
"""
Problem:
RuntimeError: received 0 items of ancdata
https://github.com/pytorch/pytorch/issues/973
"""
torch.multiprocessing.set_sharing_strategy('file_system')


class BinaryClassificationTrainer(BaseTrainer):
    def forward_pass(self, data, **kwargs):
        x = data['img']
        y = data['evidence']
        res = self.net(x)
        loss = F.binary_cross_entropy_with_logits(res['pred'], y)
        return {
            'x': x,
            'y': y,
            **data,
            'pred': res['pred'],
            'pred_seg': res['seg'],
            'loss': loss,
            'n': len(x),
        }

    @classmethod
    def make_default_callbacks(cls):
        return super().make_default_callbacks() + [
            MovingAvgCb(['loss']), ReportLRCb()
        ]


def make_net(backbone):
    return partial(Pylon, backbone=backbone)


def train(name,
          bs=64,
          dev='cuda:0',
          seed=0,
          images_path='data/images',
          backbone='resnet50',
          lr=1e-4,
          lr_term=1e-6,
          patience=1,
          n_worker=16,
          interpolation='cubic'):
    set_seed(seed)
    train_transform = make_transform('common',
                                     size=256,
                                     rotate=90,
                                     brightness=0.5,
                                     contrast=0.5,
                                     min_size=0.7,
                                     interpolation=interpolation)
    eval_transform = make_transform('eval',
                                    size=256,
                                    interpolation=interpolation)

    dataset = ChestXRay14OriginalSplit(images_path=images_path,
                                       train_transform=train_transform,
                                       eval_transform=eval_transform)

    dataset_seg = ChestXRay14Segment(images_path=images_path,
                                     eval_transform=eval_transform)

    cls_names = dataset.test_data.i_to_l

    def make_loader(dataset, shuffle=False, **kwargs):
        return ConvertLoader(
            DataLoader(
                dataset,
                batch_size=bs,
                num_workers=n_worker,
                shuffle=shuffle,
                multiprocessing_context=(mp.get_context('fork')
                                         if n_worker > 0 else None),
                **kwargs,
            ),
            device=dev,
        )

    callbacks = BinaryClassificationTrainer.make_default_callbacks() + [
        ValidateCb(
            make_loader(dataset.val_data),
            name='val',
            callbacks=[AvgCb('loss'),
                       AUROCCb(cls_names=cls_names)],
        ),
        ValidateCb(
            make_loader(dataset.test_data),
            name='test',
            callbacks=[AvgCb('loss'),
                       AUROCCb(cls_names=cls_names)],
        ),
        ValidateCb(
            make_loader(dataset_seg.segment_only_data, collate_fn=collate_fn),
            name='test',
            callbacks=[PointLocalizationCb(cls_names=cls_names)],
        ),
        LRReducePlateauCb('val_loss', mode='min', patience=patience),
        TerminateLRCb(lr_term),
        LiveDataframeCb(f'csv/{name}/{seed}.csv'),
        AutoResumeCb(f'save/{name}/{seed}',
                     keep_best=True,
                     metric='val_loss',
                     metric_best='min',
                     resume=True),
    ]

    def make_opt(net):
        return optim.Adam(net.parameters(), lr=lr)

    cls = apex_trainer_mask(BinaryClassificationTrainer)
    trainer = cls(
        make_net(backbone=backbone),
        make_opt,
        dev,
        callbacks,
    )
    trainer.train(
        make_loader(dataset.train_data, shuffle=True),
        n_max_ep=40,
    )


if __name__ == "__main__":
    train(
        name='pylon',
        dev='cuda:0',
        bs=64,
        seed=0,
        n_worker=4,
        interpolation='cubic',
    )
