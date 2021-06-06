import pickle

from .types import *

__all__ = ['NumpyWriter', 'numpy_reader']

class NumpyWriter:
    def __init__(self, path, n_max_width=1000):
        self.path = path
        self.n_max_width = n_max_width

        self._buffer = {}
        self.f = open(self.path, 'ab+')

    def write_hist(self, name, val, i_itr):
        if isinstance(val, Tensor):
            val = val.detach().cpu().numpy()
        assert isinstance(val, np.ndarray)
        # flatten
        val = val.reshape(-1)
        # reduce the val size
        if len(val) > self.n_max_width:
            val = np.random.choice(val, size=self.n_max_width)
        # write the array
        self._buffer[name] = val
        if 'i_itr' in self._buffer:
            assert self._buffer['i_itr'] == i_itr, "should forgot to flush"
        self._buffer['i_itr'] = i_itr

    def flush(self):
        """save the buffer to file"""
        # not empty
        if len(self._buffer.keys()) > 0:
            pickle.dump(self._buffer, self.f)
            self._buffer = {}
            self.f.flush()

    def close(self):
        self.f.close()

def numpy_reader(path):
    with open(path, 'rb') as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                break
