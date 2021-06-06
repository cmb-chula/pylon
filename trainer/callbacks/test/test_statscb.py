from mlkit.start import *
from mlkit.trainer import *
from mlkit.trainer.callbacks.base_cb import *
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
    1000,
    callbacks=trainer.make_default_callbacks() + [
        AutoResumeCb(f'{dirname}/save', n_save_cycle=100),
        ValidateCb(FlattenLoader(data.test_loader(32)), 100)
    ]
)
print(df)
