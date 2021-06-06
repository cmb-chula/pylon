import json
import multiprocessing as mp
import os
from dataclasses import dataclass
from typing import Tuple

import torch


@dataclass
class Env:
    namespace: str = ''
    cuda: Tuple[int] = tuple(range(torch.cuda.device_count()))
    global_lock: int = 1
    num_workers: int = mp.cpu_count()

    @classmethod
    def from_json(cls, path):
        with open(path) as f:
            env = json.load(f)

        new = Env()
        for k, v in env.items():
            new.__dict__[k] = v
        return new


def read_env_file():
    for file in ENVFILES:
        if os.path.exists(file):
            return Env.from_json(file)
    return Env()


ENVFILES = [
    'mlkitenv.json',
    os.path.expanduser('~/mlkitenv.json'),
]
ENV = read_env_file()
