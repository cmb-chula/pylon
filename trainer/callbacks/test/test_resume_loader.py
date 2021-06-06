from mlkit.start import *
from mlkit.vision.data.imagenet_lmdb import Imagenet2012
from mlkit.trainer.start import *
from mlkit.trainer.apex_trainer import *
from torchvision.models import mobilenet_v2

dataset = Imagenet2012(32, 'cpu', 1)

def make_opt(net):
    return optim.SGD(net.parameters(), lr=0.1)

dirname = os.path.dirname(__file__)

trainer = SimpleTrainer(mobilenet_v2, make_opt, 'cpu')
trainer.train(dataset.train_loader(False), 2, callbacks=trainer.make_default_callbacks() + [
    AutoResumeCb(os.path.join(dirname, 'tmp'), 100),
    ResumeLoader(),
])
