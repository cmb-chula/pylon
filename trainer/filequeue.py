import os
import time
from contextlib import ContextDecorator

from .envfile import ENV
from .filelock import FileLock, nullcontext


def global_queue(n=None,
                 delay=3.0,
                 verbose=True,
                 enable=True,
                 namespace: str = None):
    """using a global lock file shared across the user"""
    if n is None:
        n = ENV.global_lock or 1
    if namespace is None:
        namespace = ENV.namespace

    if enable:
        return FileQueue(n=n,
                         delay=delay,
                         verbose=verbose,
                         file_prefix=f'mlkit{namespace}.queue')
    else:
        return nullcontext()


def wait_global_queue(n=None, delay=3.0, verbose=True, enable=True):
    """wait for at least one of the locks to be available but not acquiring it"""
    with global_queue(n=n, delay=delay, verbose=verbose, enable=enable):
        return


class FileQueue(ContextDecorator):
    """
    Args:
        n: number of total lock files
    """
    def __init__(self,
                 n=1,
                 delay=1.0,
                 verbose=True,
                 dirname='~',
                 file_prefix='mlkit.queue'):
        self.dirname = os.path.expanduser(dirname)
        self.n = n
        self.delay = delay
        self.file_prefix = file_prefix
        self.verbose = verbose

        self.is_locked = False
        self.lockfile = None

    def has_right(self):
        return FileLock(os.path.join(self.dirname, 'mlkit.alloc.queue'),
                        delay=0.1,
                        verbose=False)

    def filename(self, i):
        return f'{self.file_prefix}.{i}'

    def inv_filename(self, filename):
        return int(filename.split('.')[-1])

    def discover(self):
        """returns all queue files"""
        ids = []
        for f in os.listdir(self.dirname):
            if self.file_prefix in f:
                ids.append(self.inv_filename(f))
        return sorted(ids)

    def acquire(self):
        while True:
            # queing
            with self.has_right():
                if self.lockfile is None:
                    ids = self.discover()
                    if len(ids) == 0:
                        next_id = 0
                    else:
                        next_id = max(ids) + 1
                    # create the queue file
                    self.lockfile = os.path.join(self.dirname,
                                                 self.filename(next_id))
                    os.close(os.open(self.lockfile, os.O_CREAT))
                    if self.verbose:
                        print(f'Queue file {self.lockfile} acquired')

                # activate
                if self.lockfile is not None:
                    ids = self.discover()
                    our_id = self.inv_filename(self.lockfile)
                    # if we are in top n
                    order = ids.index(our_id)
                    if order < self.n:
                        # locked
                        if self.verbose:
                            print(f'Queue {our_id} now runs ...')
                        self.is_locked = True
                        return

            if not self.is_locked:
                time.sleep(self.delay)

    def release(self):
        if self.lockfile is not None:
            try:
                os.unlink(self.lockfile)
                if self.verbose: print(f'Queue file {self.lockfile} released')
            except Exception as e:  # ignore errors
                print(f'error releasing lock file {self.lockfile}:', e)
            self.is_locked = False
            self.lockfile = None

    def __enter__(self):
        if not self.is_locked:
            self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        if self.is_locked:
            self.release()

    def __del__(self):
        self.release()


if __name__ == "__main__":
    with FileQueue(verbose=True):
        print('aoeu')
