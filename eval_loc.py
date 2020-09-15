from dataset_segment import *
from train import BinaryClassificationTrainer, make_net
from trainer.all import *
from trainer.callbacks.all import *
from utils.csv import *
from utils.loader import *
from utils.localization import *


def eval_localization(name,
                      seed,
                      make_net,
                      mode='best',
                      images_path='data/images',
                      bs=64,
                      dev='cuda:0',
                      size=256,
                      interpolation='cubic',
                      n_worker=4):
    eval_transform = make_transform('eval',
                                    size=size,
                                    interpolation=interpolation)
    dataset = ChestXRay14Segment(images_path, eval_transform=eval_transform)

    loader = ConvertLoader(
        DataLoader(dataset.segment_only_data,
                   batch_size=bs,
                   collate_fn=collate_fn,
                   num_workers=n_worker), dev)

    trainer = BinaryClassificationTrainer(make_net, None, dev)
    trainer.load(f'save/{name}/{seed}/{mode}')

    # metric
    metric = PointLocalizationCb(cls_names=dataset.segment_only_data.i_to_l)
    callbacks = [
        ProgressCb(desc='predict'),
        metric,
    ]
    # predict
    predictor = BasePredictor(trainer, callbacks, ['seg', 'has', 'pred_seg'])
    predictor.predict(loader)

    cls_names = [
        'Atelectasis',
        'Cardiomegaly',
        'Effusion',
        'Infiltration',
        'Mass',
        'Nodule',
        'Pneumonia',
        'Pneumothorax',
    ]

    # write as df
    res = metric.last_hist
    df = {k: [] for k in cls_names}
    df['micro'] = []
    for cls in cls_names:
        val = res[f'point_acc_{cls}']
        df[cls].append(val)
    df['micro'].append(res[f'point_acc_micro'])
    df = pd.DataFrame(df)
    dirname = f'eval_loc/{name}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    df.to_csv(f'{dirname}/{seed}.csv', index=False)
    # group the seeds, it will be correct with the last seed
    group_seeds(dirname)
    return res


if __name__ == "__main__":
    name = 'pylon'
    for seed in [0]:
        eval_localization(
            name=name,
            seed=seed,
            make_net=make_net('resnet50'),
            interpolation='cubic',
        )
