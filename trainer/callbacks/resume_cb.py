import time
from ..save import *
from ..seed import *
from .base_cb import *
import shutil
import json


class AutoResumeCb(Callback):
    """auto save and resume for all callbacks and model, with multiple checkpoints
    Args:
        n_save_cycle: -1 to ignore, default: n_ep_itr
        resume: using save mode, without resuming
        resume_from: 'last', 'best', or iteration number as int
        keep_best: keep the best checkpoint
        keep_all: never destroy previous checkpoints
    """
    # we hope to get the stats at the very end of batch end"""
    _order = 1000

    def __init__(
            self,
            dirname: str,
            n_itr_cycle: int = None,
            n_ep_cycle: float = None,
            resume: bool = True,
            resume_from='last',
            n_keep: int = 0,
            keep_best=False,
            keep_fn=None,
            metric: str = 'val_acc',
            metric_best: str = 'max',  # or min
            map_location=None,
            extras=None,
            verbose=False,
            save_abrupt_end=True,
            return_best_model=False,
            **kwargs):
        super().__init__(**kwargs)
        self.dirname = dirname
        self.n_itr_cycle = n_itr_cycle
        self.n_ep_cycle = n_ep_cycle
        self.resume = resume
        self.resume_from = resume_from
        self.n_keep = n_keep
        self.keep_best = keep_best
        if keep_fn is not None:
            assert callable(keep_fn), 'keep function must be callable'
        self.keep_fn = keep_fn
        self.metric = metric
        self.metric_best = metric_best
        self.map_location = map_location
        self.extras = extras
        self.verbose = verbose
        self.save_abrupt_end = save_abrupt_end
        self.return_best_model = return_best_model

        # keep tracks used in best
        self._state['history'] = {'i_itr': [], 'metric': []}
        self._state['excludes'] = []

    # should resume before normal loop
    @set_order(90)
    def on_train_begin(self, trainer, callbacks, n_ep_itr, **kwargs):
        if self.n_itr_cycle is None:
            if self.n_ep_cycle is not None:
                self.n_itr_cycle = int(self.n_ep_cycle * n_ep_itr)
            else:
                # default to 1 ep
                self.n_itr_cycle = n_ep_itr
        if self.resume:
            self._load(trainer, callbacks)
            self._start_itr = trainer.i_itr

    def on_batch_end(self, i_itr, trainer, callbacks, **kwargs):
        if self.n_itr_cycle > 0 and i_itr % self.n_itr_cycle == 0:
            self._save(i_itr, trainer, callbacks, **kwargs)

    def on_train_end(self, i_itr, trainer, callbacks, **kwargs):
        if i_itr > 1:
            # it must have something to save
            if i_itr == self._get_latest_itr():
                # do nothing, no need to save
                return
            self._save(i_itr,
                       trainer,
                       callbacks,
                       ignore_history=True,
                       **kwargs)

        if self.return_best_model:
            # load the best model at the end
            self._load_best(trainer)

    def on_abrupt_end(self, i_itr, trainer, callbacks, **kwargs):
        if self.save_abrupt_end:
            if i_itr > 1:
                # it must have something to save
                last_itr = self._get_latest_itr()
                if last_itr is not None and i_itr <= last_itr:
                    # abrupt end will not destroy the latest dir
                    return
                self._save(i_itr,
                           trainer,
                           callbacks,
                           ignore_history=True,
                           **kwargs)

        if self.return_best_model:
            # load the best model at the end
            self._load_best(trainer)

    def _load_best(self, trainer):
        """load best iteration to the trainer"""
        best_itr = self._get_best_itr()
        if best_itr is None:
            print('warning: there is no best model to load')
            return
        best_path = os.path.join(self.dirname, str(best_itr))
        trainer.load(best_path)

    def _load(self, trainer, callbacks):
        """load a checkpoint, decides where to load"""
        if self.resume_from == 'last':
            itr = self._get_latest_itr()
        elif self.resume_from == 'best':
            itr = 'best'
        else:
            itr = self.resume_from
        if itr is None:
            # nothing to resume from
            return
        if self.verbose: print(f'resuming from {itr} ...')
        dirname = os.path.join(self.dirname, str(itr))
        load_all(
            dirname,
            trainer,
            callbacks,
            map_location=self.map_location,
            verbose=self.verbose,
            load_rng=True,
        )

    def _get_metric(self, callbacks):
        v = None
        all_keys = []
        for cb in callbacks:
            if isinstance(cb, StatsCallback):
                all_keys += list(cb.stats.keys())
                if self.metric in cb.stats:
                    v = cb.stats[self.metric]
                    break
        assert v is not None, f"metric {self.metric} must be present for the save, you must need to set the cycles correctly, there are {all_keys}"
        return v

    def _save(self,
              i_itr,
              trainer,
              callbacks,
              ignore_history: bool = False,
              **kwargs):
        """save a checkpoint, decides where to save"""
        try:
            if self.keep_best:
                metric = self._get_metric(callbacks)
                # jot the metrics (must be before)
                self._state['history']['i_itr'].append(i_itr)
                self._state['history']['metric'].append(metric)
        except Exception:
            # ignore history when it is ended by train_end
            if ignore_history:
                pass
            else:
                # raise normally
                raise

        # save a checkpoint in a direction with the iteration name
        last_dir = str(i_itr)
        dirname = os.path.join(self.dirname, last_dir)
        if self.verbose: print(f'saving {last_dir} ...')
        save_all(dirname,
                 trainer,
                 callbacks,
                 verbose=self.verbose,
                 extras=self.extras)

        # for legacy
        if 'excludes' not in self._state:
            self._state['excludes'] = []

        # exclude from a keep function
        if self.keep_fn is not None:
            if self.keep_fn(i_itr=i_itr, trainer=trainer, **kwargs):
                # exclude this i_itr
                self._state['excludes'].append(str(i_itr))

        exclude = [last_dir] + self._state['excludes']
        # if keep best, point the best directory to this
        if self.keep_best:
            best_dir = self._symlink_best()
            exclude += [best_dir]

        # keep only n last (also the best if available)
        self._prune(i_itr, exclude)

        # make symlink to the last dir
        self._symlink_latest()

    def _prune(self, i_itr, exclude: List[str]):
        prune(i_itr, exclude, self.n_keep, self.dirname, verbose=self.verbose)

    def _get_best_itr(self):
        if len(self._state['history']['i_itr']) > 0:
            metric = self._state['history']['metric']
            if self.metric_best == 'min':
                arg_best = np.argmin(metric)
            elif self.metric_best == 'max':
                arg_best = np.argmax(metric)
            else:
                raise NotImplementedError()
            tgt_itr = self._state['history']['i_itr'][arg_best]
            return tgt_itr
        else:
            return None

    def _get_latest_itr(self):
        """look in the directory find the largest number"""
        return get_latest_itr(self.dirname)

    def _symlink_best(self):
        return symlink(self.dirname, self._get_best_itr(), 'best')

    def _symlink_latest(self):
        return symlink(self.dirname, self._get_latest_itr(), 'latest')


# alias
AutoResume2Cb = AutoResumeCb


def combine_keep_fn(fns):
    def fn(*args, **kwargs):
        # any
        out = False
        for fn in fns:
            out |= fn(*args, **kwargs)
        return out

    return fn


class ExcludeOnLRChange:
    """
    Args:
        trainer: to get the initial learning rate
    """
    def __init__(self, verbose=False):
        self.current_lr = None
        self.verbose = verbose

    def get_lr(self, trainer):
        return trainer.opt.param_groups[0]['lr']

    def __call__(self, trainer, **kwargs):
        """shall exclude when the learning rate changes"""
        new_lr = self.get_lr(trainer)
        if self.current_lr is None:
            self.current_lr = new_lr

        if self.verbose: print('new lr:', new_lr)
        if new_lr != self.current_lr:
            if self.verbose: print('lr changes!')
            self.current_lr = new_lr
            return True
        return False


class SaveEvery:
    """
    Args:
        trainer: to get the initial learning rate
    """
    def __init__(self, n_itr):
        self.n_itr = n_itr

    def __call__(self, i_itr, **kwargs):
        """shall exclude when the learning rate changes"""
        if self.n_itr > 0:
            return i_itr % self.n_itr == 0
        else:
            return False


def iterate_cb(callbacks):
    cnt = defaultdict(lambda: 0)
    for cb in callbacks:
        name = type(cb).__name__
        _name = f'{name}_{cnt[name]}'
        yield cb, _name
        cnt[name] += 1


def save_random_state(path):
    safe_torch_save(get_random_state(), path)


def load_random_state(path):
    set_random_state(torch.load(path))


def prune(i_itr, exclude: List[str], n_keep, dirname, verbose):
    # list dirs to be remove
    exclude += ['best', 'latest']  # remove 'best' from the list
    paths = set(os.listdir(dirname))
    paths = paths - set(exclude)
    # sort by itr
    nums = []
    for x in paths:
        try:
            nums.append(int(x))
        except Exception:
            pass
    nums = sorted(nums, reverse=True)
    before = [x for x in nums if x <= i_itr]
    after = [x for x in nums if x > i_itr]
    # we remove those after
    # keep n_keep in before
    # remove all after
    removes = before[n_keep:] + after

    # try remove the latest dir
    # should make symlink after prune
    # this is to make sure that the symlink is always the latest
    try:
        os.remove(os.path.join(dirname, 'latest'))
    except OSError:
        pass

    for itr in removes:
        if verbose: print(f'pruning {itr}')
        _dirname = os.path.join(dirname, str(itr))
        shutil.rmtree(_dirname)


def get_latest_itr(dirname):
    """look in the directory find the largest number"""
    if not os.path.exists(dirname):
        return None
    itrs = sorted(
        [int(p) for p in os.listdir(dirname) if p not in ('best', 'latest')])
    if len(itrs) == 0:
        return None
    return itrs[-1]


def symlink(dirname, tgt_itr, name):
    """actually save the chekcpoint given the location"""
    # try remove the current symlink
    sym_dir = os.path.join(dirname, name)
    try:
        os.remove(sym_dir)
    except OSError:
        pass

    if tgt_itr is None:
        return None

    # create a new symlink to the best
    os.symlink(
        str(tgt_itr),
        name,
        target_is_directory=True,
        dir_fd=os.open(dirname,
                       os.O_RDONLY),  # important to make the link visible
    )
    return str(tgt_itr)


def save_all(dirname, trainer, callbacks, extras=None, verbose=False):
    """actually save the chekcpoint given the location"""
    start_time = time.time()
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    # write extras
    if extras is not None:
        with open(os.path.join(dirname, 'extras.json'), 'w') as f:
            try:
                json.dump(extras, f)
            except Exception as e:
                print('error during writing json:', e)
    # save rng state
    path = os.path.join(dirname, 'rng.pkl')
    save_random_state(path)
    # save trainer
    trainer.save(dirname)
    # save callbacks
    path = os.path.join(dirname, 'callbacks')
    try:
        for cb, name in iterate_cb(callbacks):
            cb_path = os.path.join(path, f'{name}.pkl')
            cb.save(cb_path)
    except Exception:
        print(f'unable to save {cb}')
        raise
    if verbose:
        print(f'saving took {time.time() - start_time:.2f} seconds')


def load_all(dirname,
             trainer,
             callbacks,
             map_location=None,
             verbose=False,
             load_rng=True):
    """actually load the chekcpoint given the location"""
    if os.path.exists(dirname):
        start_time = time.time()
        # set default map location to the trainer's device
        if map_location is None:
            map_location = trainer.device

        # load random state
        if load_rng:
            try:
                path = os.path.join(dirname, 'rng.pkl')
                load_random_state(path)
            except FileNotFoundError:
                print('random state file not found ... skipping')
        # load trainer
        if os.path.exists(os.path.join(dirname, 'model.pkl')):
            # new version
            trainer.load(dirname)
        else:
            # old version
            path = os.path.join(dirname, 'trainer.pkl')
            trainer.load(path)
        # load callbacks
        path = os.path.join(dirname, 'callbacks')
        for cb, name in iterate_cb(callbacks):
            try:
                cb_path = os.path.join(path, f'{name}.pkl')
                cb.load(cb_path, map_location=map_location)
            except FileNotFoundError:
                print(f'callback {name}' f' file not found ... skipping')
            except Exception as e:
                if os.environ.get('MLKIT_RESUME_IGNORE', False):
                    print(
                        f'callback {name} (ignore due to MLKIT_RESUME_IGNORE)'
                        f' error {e} ... skipping')
                else:
                    raise e

        if verbose:
            print(f'resuming took {time.time() - start_time:.2f} seconds')
    else:
        raise FileNotFoundError(f'load failed: {dirname}')
