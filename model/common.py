from trainer.start import *


@dataclass
class ModelReturn:
    pred: Tensor
    pred_seg: Tensor
    loss: Optional[Tensor]
    loss_pred: Optional[Tensor]
    loss_bbox: Optional[Tensor]
