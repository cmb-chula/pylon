import multiprocessing as mp
import queue
import threading as th

import torch

from .loader_base import BaseLoaderWrapper


class Prefetcher(BaseLoaderWrapper):
    """prefetch data from the loader, it will automatically restart the loader
    doesn't work with pin_memory=True"""
    class Worker(mp.Process):
        """this will diligently call for data until the queue is full.
        This is implemented as a process because we need to be able to stop it."""
        def __init__(self, loader, q: mp.Queue):
            super().__init__()
            self.loader = loader
            self.q = q

        def run(self):
            """loop and put the data into the queue"""
            while True:
                for data in self.loader:
                    self.q.put(data)
                # end of the dataset
                self.q.put(None)

    def __init__(self, loader, n_prefetch):
        super().__init__(loader)
        self.q = mp.Queue(maxsize=n_prefetch)
        self.worker = None
        self._stats = {}

    def __del__(self):
        self.stop_worker()

    def start_worker(self):
        """create and start the worker"""
        self.stop_worker()
        self.worker = Prefetcher.Worker(self.loader, self.q)
        self.worker.start()

    def stop_worker(self):
        if self.worker is not None:
            self.worker.kill()
            self.worker.join()
            self.worker.close()

    def stats(self):
        return self._stats

    def __iter__(self):
        if self.worker is None:
            self.start_worker()

        while True:
            if not self.worker.is_alive():
                print('prefetcher: restart worker')
                self.start_worker()

            self._stats['prefetch'] = self.q.qsize()
            try:
                data = self.q.get(timeout=15)
            except queue.Empty:
                print('prefetcher: restart worker')
                self.start_worker()
                continue
            # end of the dataset
            if data is None:
                break
            yield data


class StoppableThread(th.Thread):
    """
    A thread that has a 'stop' event.
    from: Tensorpack
    """
    def __init__(self, evt=None):
        """
        Args:
            evt(threading.Event): if None, will create one.
        """
        super(StoppableThread, self).__init__()
        if evt is None:
            evt = th.Event()
        self._stop_evt = evt

    def stop(self):
        """ Stop the thread"""
        self._stop_evt.set()

    def stopped(self):
        """
        Returns:
            bool: whether the thread is stopped or not
        """
        return self._stop_evt.isSet()

    def queue_put_stoppable(self, q, obj, timeout=1):
        """ Put obj to queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                q.put(obj, timeout=timeout)
                break
            except queue.Full:
                pass

    def queue_get_stoppable(self, q, timeout):
        """ Take obj from queue, but will give up when the thread is stopped"""
        while not self.stopped():
            try:
                return q.get(timeout=timeout)
            except queue.Empty:
                pass


class PrefetcherThread(BaseLoaderWrapper):
    """prefetch data from the loader, 
    works in most casess, but will not automatically restart the dataloader (because it happens to lead to deadlocks)"""
    class Worker(StoppableThread):
        """this will diligently call for data until the queue is full"""
        def __init__(self, loader, q: queue.Queue, device):
            super().__init__()
            self.loader = loader
            self.q = q
            self.device = device
            self.daemon = True

        def run(self):
            # the default default is not inherited from the main thread
            torch.cuda.set_device(self.device)
            try:
                # put data to saturate the queue
                for data in self.loader:
                    if self.stopped():
                        return
                    self.queue_put_stoppable(self.q, data)
                # end of the dataset
                self.queue_put_stoppable(self.q, None)
            except Exception:
                if self.stopped():
                    pass  # skip duplicated error messages
                else:
                    raise
            finally:
                self.stop()

    def __init__(self, loader, n_prefetch, device):
        super().__init__(loader)
        self.q = queue.Queue(maxsize=n_prefetch)
        self.device = device
        self.worker = None
        self._stats = {}

    def stats(self):
        return self._stats

    def __iter__(self):
        self.worker = PrefetcherThread.Worker(
            self.loader, self.q, device=self.device)
        self.worker.start()
        while True:
            self._stats['prefetch'] = self.q.qsize()
            data = self.q.get()
            # end of the dataset
            if data is None:
                break
            yield data
        self.worker.join()
