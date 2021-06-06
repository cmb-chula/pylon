from multiprocessing.dummy import Pool
from multiprocessing import get_context
from tqdm.autonotebook import tqdm


def multiprocess_map(fn,
                     args,
                     num_workers: int = None,
                     progress: bool = False,
                     debug: bool = False):
    if num_workers is None:
        num_workers = len(args)
    if debug:
        for each in [(fn, arg) for arg in args]:
            _call_fn_under_process(each)
    else:
        with Pool(num_workers) as pool:
            iter = pool.imap_unordered(_call_fn_under_process,
                                       [(fn, arg) for arg in args])
            if progress:
                iter = tqdm(iter, total=len(args))
            for each in iter:
                pass


def _call_fn_under_process(fn_arg):
    fn, arg = fn_arg
    ctx = get_context('spawn')
    proc = ctx.Process(target=fn, args=(arg, ))
    proc.start()
    proc.join()


def _test(arg):
    print('arg:', arg)


if __name__ == '__main__':
    multiprocess_map(_test, [1, 2, 3, 4], 2)
