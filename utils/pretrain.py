import os
from typing import Dict

import torch
from torch import nn
from trainer.start import *

PRETRAIN_NAMES = {
    'baseline,nih,256': 'pretrain/baseline,nih,256.pkl',
    'pylon,nih,256': 'pretrain/pylon,nih,256.pkl',
}


@dataclass
class PretrainConfig(BaseConfig):
    pretrain_name: str = None
    prefix: str = None
    ignore: str = None
    path: str = None

    @property
    def name(self):
        name = ''
        if self.pretrain_name is not None:
            name += f'w{self.pretrain_name}'
            if self.ignore is not None:
                name += f'-ign({self.ignore})'
        return name


def prefix_state_dict(state: Dict, prefix):
    out = {}
    for k, v in state.items():
        out[f'{prefix}{k}'] = v
    return out


def keep_only_matched_sizes(state: Dict, target_state: Dict):
    keys = set(state.keys()) & set(target_state.keys())
    out = {}
    for k in keys:
        if state[k].shape == target_state[k].shape:
            out[k] = state[k]
    return out


def ignore_state_dict(state: Dict, prefix):
    if prefix is None:
        return state
    out = {}
    for k, v in state.items():
        if k[:len(prefix)] != prefix:
            out[k] = v
    return out


def get_pretrain_path(config_name: str,
                      base_dir='save',
                      postfix='checkpoints/best/model.pkl'):
    return os.path.join(base_dir, config_name, postfix)


def load_pretrain(
    conf: PretrainConfig,
    target: nn.Module,
):
    """
    Args:
        conf:
        target: target model
    """
    if conf.path is None:
        path = PRETRAIN_NAMES[conf.pretrain_name]
    else:
        path = conf.path
    state = torch.load(path, map_location='cpu')
    if conf.prefix is not None:
        state = prefix_state_dict(state, conf.prefix)
    state = ignore_state_dict(state, conf.ignore)
    state = keep_only_matched_sizes(state, target.state_dict())
    err = target.load_state_dict(state, strict=False)
    print(f'err:', err)
