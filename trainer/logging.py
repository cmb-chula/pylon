from contextlib import contextmanager, nullcontext
import os
from datetime import datetime
import uuid
import sys


@contextmanager
def redirect_to_file(dirname='logs',
                     mode='w',
                     redirect_stderr=True,
                     redirect_stdout=True,
                     enable=True):
    if not enable:
        with nullcontext():
            yield
        return

    if redirect_stderr:
        old_stderr = sys.stderr
    if redirect_stdout:
        old_stdout = sys.stdout
    if not os.path.exists(dirname):
        os.makedirs(dirname)
    now = datetime.now()
    rand = uuid.uuid4().hex
    date_time = now.strftime("%m-%d-%Y_%H:%M:%S")
    path = os.path.join(dirname, f'log_{date_time}_{rand[:5]}.txt')
    print(f'logging to {path} ...')
    # change the std
    file = open(path, mode)
    if redirect_stderr:
        sys.stderr = file
    if redirect_stdout:
        sys.stdout = file

    yield

    if redirect_stderr:
        sys.stderr = old_stderr
    if redirect_stdout:
        sys.stdout = old_stdout
