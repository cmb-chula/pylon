from utils.csv import *

from .base_cb import *
from tqdm.autonotebook import tqdm


class ReportItrCb(StatsCallback):
    def on_batch_begin(self, i_itr, i_ep, p_ep, **kwargs):
        self.bar_and_hist({'i_itr': i_itr, 'i_ep': i_ep, '%ep': p_ep})


class ProgressCb(Callback):
    """call and collect stats from StatsCallback
    
    Args:
        destroy: should destroy on train end? should be False with validate loop

    """
    _order = 1000  # need to wait for all stats

    def __init__(self, desc='train', destroy=True, **kwargs):
        super().__init__(**kwargs)
        self.desc = desc
        self.progress = None
        self.destroy = destroy

    def update(self, callbacks, i_itr):
        stats = {}
        for cb in callbacks:
            if isinstance(cb, StatsCallback):
                stats.update(cb.stats)

        # prevent 1e4 for int
        for k, v in stats.items():
            if isinstance(v, int):
                stats[k] = str(v)

        # don't show i_itr
        if 'i_itr' in stats:
            del stats['i_itr']

        # set postfix must not force the refresh
        self.progress.set_postfix(stats, refresh=False)
        self.progress.update(i_itr - self.progress.n)

    def _same_tqdm(self, bar, n_max_itr, i_itr):
        same = (bar.total == n_max_itr)  # and (bar.n == i_itr - 1)
        return same

    def on_train_begin(self, n_max_itr, **kwargs):
        self.progress = tqdm(total=n_max_itr,
                             desc=self.desc,
                             mininterval=0.1,
                             miniters=1)

    def on_batch_end(self, callbacks, i_itr, **kwargs):
        self.update(callbacks, i_itr)

    def on_train_end(self, callbacks, i_itr, **kwargs):
        if self.progress is not None:
            self.update(callbacks, i_itr)
            if self.destroy:
                self.progress.close()

    def on_abrupt_end(self, **kwargs):
        if self.progress is not None:
            if self.destroy:
                self.progress.close()


class LiveDataframeCb(StatsCallback):
    """pulls data from callbacks and write it into CSV in real-time"""
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.writer = None

    def on_train_begin(self, trainer, callbacks, i_itr, n_max_itr, **kwargs):
        if i_itr == n_max_itr:
            # the job is already finished!
            return

        # rewrite all to the file
        df = StatsCallback.combine_callbacks(callbacks)
        self.writer = FastCSVWriter(self.path)
        if df is not None:
            self.writer.write_df(df)

    def write(self, callbacks, i_itr):
        row = {}
        for cb in callbacks:
            if isinstance(cb, StatsCallback):
                # check if the latest entry is the one to be logged
                if len(cb.hist['i_itr']) > 0 and cb.hist['i_itr'][-1] == i_itr:
                    for k, v in cb.hist.items():
                        row.update({k: v[-1]})

        # self.writer.writekvs(row)
        self.writer.write(row)

    # after normal callbacks
    @set_order(110)
    def on_batch_end(self, callbacks, i_itr, **kwargs):
        self.write(callbacks, i_itr)

    @set_order(110)
    def on_train_end(self, callbacks, i_itr, **kwargs):
        if self.writer is not None:
            self.write(callbacks, i_itr)
            self.writer.close()


class ReportLRCb(StatsCallback):
    """
    problem: this runs "before" autoresume making it a bit ugly.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def on_batch_begin(self, trainer, i_itr, **kwargs):
        lrs = []
        for g in trainer.opt.param_groups:
            lrs.append(g['lr'])

        info = {'i_itr': i_itr, 'lr': lrs[0]}
        if len(lrs) > 1:
            info.update({f'lr{i+1}': lr for i, lr in enumerate(lrs[1:])})
        self.bar_and_hist(info)


class ReportWeightNormCb(StatsCallback):
    def __init__(self, name='weight_norm', use_histogram=False, **kwargs):
        super().__init__(**kwargs)
        self.name = name
        self.use_histogram = use_histogram

    def on_backward_end(self, trainer, i_itr, **kwargs):
        if self.use_histogram:
            self.write_hist(
                f'{self.name}/hist', lambda: nn.utils.parameters_to_vector(
                    iter_opt_params(trainer.opt)), i_itr)
        self.bar_and_hist({
            'i_itr':
            i_itr,
            self.name:
            lambda: many_l2_norm(*list(iter_opt_params(trainer.opt)))
        })


def list_name(net, prefix=''):
    """traverse the module to get each layer and their names, 
    it outputs in the same order as net.parameters()"""
    has_child = False

    out = []
    for i, m in enumerate(net.children()):
        has_child = True
        m_name = f'{prefix}/{i}_{m.__class__.__name__}'
        out += list_name(m, m_name)

    if not has_child:
        for name, p in net.named_parameters():
            out.append((f'{prefix}/{name}', p))

    return out
