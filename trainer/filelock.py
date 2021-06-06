import contextlib
import errno
import multiprocessing as mp
import os
import time
from contextlib import ContextDecorator

from .envfile import ENV


def global_lock(n=None, delay=3.0, verbose=True, enable=True):
    """using a global lock file shared across the user"""
    if n is None:
        n = ENV.global_lock

    if enable:
        return FileLock(None,
                        path=os.path.expanduser('~/mlkit.lock'),
                        n=n,
                        delay=delay,
                        verbose=verbose)
    else:
        return nullcontext()


def wait_global_lock(n=None, delay=3.0, verbose=True, enable=True):
    """wait for at least one of the locks to be available but not acquiring it"""
    with global_lock(n=n, delay=delay, verbose=verbose, enable=enable):
        return


@contextlib.contextmanager
def nullcontext():
    """a context manager than yield a null object which can be called without any result"""
    class NullCls:
        def __getattr__(self, name):
            return nullfn

    yield NullCls()


def nullfn(*args, **kwargs):
    pass


class FileLockException(Exception):
    pass


class FileLock(ContextDecorator):
    """ A file locking mechanism that has context-manager support so 
        you can use it in a with statement. This should be relatively cross
        compatible as it doesn't rely on msvcrt or fcntl for the locking.
        
        From: https://github.com/dmfrey/FileLock/blob/master/filelock/filelock.py

    Args:
        n: number of total lock files
    """
    def __init__(self, file_name, n=1, delay=1.0, verbose=True, path=None):
        self.is_locked = False

        def lockfile_path(i):
            if path is None:
                return os.path.join(
                    os.getcwd(),
                    f'{file_name}.lock.{i}')  # use working directory
            else:
                return f'{path}.{i}'

        self.all_lockfiles = [lockfile_path(i) for i in range(n)]
        self.lockfile = None
        self.delay = delay
        self.verbose = verbose

    def acquire(self):
        if self.verbose: print('Acquiring for a lockfile')
        while True:
            for lockfile in self.all_lockfiles:
                try:
                    self.fd = os.open(lockfile,
                                      os.O_CREAT | os.O_EXCL | os.O_RDWR)
                    self.lockfile = lockfile
                    self.is_locked = True
                    if self.verbose: print(f'Lockfile {lockfile} acquired')
                    return
                except OSError as e:
                    if e.errno != errno.EEXIST:
                        raise
            if not self.is_locked:
                time.sleep(self.delay)

    def release(self):
        """ Get rid of the lock by deleting the lockfile. 
            When working in a `with` statement, this gets automatically 
            called at the end.
        """
        if self.is_locked:
            try:
                os.close(self.fd)
                os.unlink(self.lockfile)
                if self.verbose: print(f'Lockfile {self.lockfile} released')
            except Exception as e:  # ignore errors
                print(f'error releasing lock file {self.lockfile}:', e)
            self.is_locked = False

    def __enter__(self):
        """ Activated when used in the with statement. 
            Should automatically acquire a lock to be used in the with block.
        """
        if not self.is_locked:
            self.acquire()
        return self

    def __exit__(self, type, value, traceback):
        """ Activated at the end of the with statement.
            It automatically releases the lock if it isn't locked.
        """
        if self.is_locked:
            self.release()

    def __del__(self):
        """ Make sure that the FileLock instance doesn't leave a lockfile
            lying around.
        """
        self.release()


if __name__ == "__main__":

    @global_lock(n=1)
    def test():
        print('test')

    test()
