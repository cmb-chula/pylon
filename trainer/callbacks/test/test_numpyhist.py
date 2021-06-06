from mlkit.start import *
from mlkit.trainer.start import *
from mlkit.vision.data import FashionMNIST

dataset = FashionMNIST('cuda')

def make_net():
    return nn.Sequential(nn.Linear(28 * 28, 100), nn.ReLU(), nn.Linear(100, 10))

def make_opt(net):
    return optim.Adam(net.parameters(), lr=1e-3)

dirname = os.path.dirname(__file__)
path = os.path.join(dirname, 'w.pkl')
if os.path.exists(path):
    os.remove(path)

trainer = BaseTrainer(make_net, make_opt, 'cuda')
trainer.train(
    FlattenLoader(dataset.train_loader(100)),
    30,
    callbacks=trainer.make_default_callbacks() + [
        NumpyWriterCb(path),
        NumpyWeightHistCb(10),
    ],
)

from mlkit.trainer.numpy_writer import *

for obj in numpy_reader(path):
    print(obj)
