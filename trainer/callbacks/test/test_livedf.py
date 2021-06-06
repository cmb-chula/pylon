from mlkit.start import *
from mlkit.trainer.start import *
from mlkit.trainer.callbacks.resume import AutoResumeCb
from mlkit.vision.data import MNIST

def net_fn():
    return nn.Sequential(nn.Linear(28 * 28, 10))

def opt_fn(net):
    return optim.SGD(net.parameters(), lr=0.1)

trainer = SimpleTrainer(net_fn, opt_fn, 'cpu')

dirname = os.path.dirname(__file__)
data = MNIST('cpu')
df = trainer.train(
    FlattenLoader(data.train_loader(32)),
    10,
    callbacks=trainer.make_default_callbacks() + [
        AutoResumeCb(f'{dirname}/save', n_save_cycle=100),
        LiveDataframeCb(os.path.join(dirname, 'df.csv')),
    ]
)
df.to_csv(os.path.join(dirname, 'df.csv'), index=False)

df = trainer.train(
    FlattenLoader(data.train_loader(32)),
    20,
    callbacks=trainer.make_default_callbacks() + [
        AutoResumeCb(f'{dirname}/save', n_save_cycle=100),
        LiveDataframeCb(os.path.join(dirname, 'df.csv')),
    ]
)
df.to_csv(os.path.join(dirname, 'df.csv'), index=False)

print(df)
