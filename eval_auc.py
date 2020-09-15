from dataset import *
from mlkit.start import *
from mlkit.trainer.start import *
from train import BinaryClassificationTrainer, make_net
from utils.csv import *


def eval_auc(name,
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
    dataset = ChestXRay14OriginalSplit(images_path=images_path,
                                       train_transform=None,
                                       eval_transform=eval_transform)
    loader = ConvertLoader(
        DataLoader(dataset.test_data,
                   batch_size=bs,
                   shuffle=True,
                   num_workers=n_worker), dev)

    trainer = BinaryClassificationTrainer(make_net, None, dev)
    trainer.load(f'save/{name}/{seed}/{mode}')

    cls_names = dataset.test_data.i_to_l
    auroc = AUROCCb(cls_names=cls_names)
    callbacks = [
        ProgressCb(desc='predict'),
        auroc,
    ]
    predictor = BasePredictor(trainer, callbacks)
    predictor.predict(loader)

    # to list
    cls_names = [cls_names[i] for i in range(len(cls_names))]
    # write as csv
    res = auroc.last_hist
    df = {k: [] for k in cls_names}
    df['micro'] = []
    df['macro'] = []
    for cls in cls_names:
        val = res[f'auroc_{cls}']
        df[cls].append(val)
    df['micro'].append(res[f'auroc_micro'])
    df['macro'].append(res[f'auroc_macro'])
    df = pd.DataFrame(df)
    dirname = f'eval_auc/{name}'
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    df.to_csv(f'{dirname}/{seed}.csv', index=False)
    # group the seeds, it will be correct with the last seed
    group_seeds(dirname)
    return res


if __name__ == "__main__":
    name = 'pylon'
    for seed in [0]:
        eval_auc(
            name=name,
            seed=seed,
            make_net=make_net('resnet50'),
            interpolation='cubic',
        )
