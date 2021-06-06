import contextlib
import os
import time
from collections import defaultdict
from itertools import count

import torch

from .envfile import ENV
from .filelock import *

CUDA_DEVICES = None
CUDA_ALLOC_FILE = os.path.expanduser('~/mlkit.alloc')

def set_cuda_devices(devices):
    global CUDA_DEVICES
    CUDA_DEVICES = devices

def lockfile(path):
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
        return fd, path
    except OSError as e:
        # it should not exist
        raise e

def get_alloc_right(verbose=False):
    """global allocation lock"""
    return FileLock(CUDA_ALLOC_FILE, delay=0.1, verbose=verbose)

def list_cuda_locks(namespace: str = ''):
    # locks are mlkit.cuda{dev}.{i}
    dirname = os.path.expanduser('~')
    locks = defaultdict(list)
    for f in os.listdir(dirname):
        if f'mlkit{namespace}.cuda' in f:
            _, dev, i = f.split('.')
            dev = int(dev[4:])  # from cuda*
            i = int(i)
            locks[dev].append(i)
    return locks

@contextlib.contextmanager
def cuda_round_robin(devices=None, verbose=False, enable=True, namespace: str = ''):
    """
    Args:
        devices: list of ints
    """
    # use default devices
    if devices is None:
        devices = CUDA_DEVICES
    if devices is None:
        if ENV.cuda is None:
            # default is all devices
            devices = range(torch.cuda.device_count())
        else:
            # using the envfile
            devices = ENV.cuda

    if devices is None or len(devices) == 0:
        raise AssertionError("no device available")

    if not enable:
        dev = f'cuda:{devices[0]}'
        torch.cuda.set_device(dev)
        yield dev
    else:
        # get alloc rights
        with get_alloc_right(verbose=verbose):
            # count the cuda locks
            locks = list_cuda_locks(namespace=namespace)
            min_dev = None
            min_cnt = float('inf')
            for dev in devices:
                cnt = len(locks[dev])
                if cnt < min_cnt:
                    min_cnt = cnt
                    min_dev = dev

            # lock the cuda file
            dirname = os.path.expanduser('~')
            for i in count(start=0):
                if i not in locks[min_dev]: break
            fd, path = lockfile(os.path.join(dirname, f'mlkit{namespace}.cuda{min_dev}.{i}'))
            if verbose: print(f'locked {path}')

        try:
            # yield
            dev = f'cuda:{min_dev}'
            torch.cuda.set_device(dev)
            yield dev
        finally:
            # remove the cuda file
            try:
                os.close(fd)
                os.unlink(path)
            except Exception:
                pass
            if verbose: print(f'released {path}')

if __name__ == "__main__":
    # print(get_alt_cuda_device(verbose=True))
    from multiprocessing.dummy import Pool

    set_cuda_devices([0, 1])

    def alloc(i):
        with cuda_round_robin() as device:
            print(f'got device:', device)
            time.sleep(1)

    pool = Pool(10)
    pool.map(alloc, range(10))
