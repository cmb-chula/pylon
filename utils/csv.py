import os

import pandas as pd
from pandas import DataFrame
from tqdm.autonotebook import tqdm


def group_seeds(dirname):
    seeds = []
    for f in os.listdir(dirname):
        num, _ = f.split('.csv')
        try:
            num = int(num)
            seeds.append(num)
        except Exception:
            pass

    out = None
    for seed in sorted(seeds):
        df = pd.read_csv(f'{dirname}/{seed}.csv')
        if out is None:
            out = df
        else:
            out = out.append(df)
    out.to_csv(f'{dirname}/all.csv', index=False)
    # calculate mean and sd
    mean = out.mean(axis=0)
    sd = out.std(axis=0)
    df = pd.DataFrame([mean, sd])
    df.to_csv(f'{dirname}/stats.csv', index=False)


class FastCSVWriter:
    def __init__(self, path, mode='w'):
        dirname = os.path.dirname(path)
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)

        self.path = path
        self.f = open(path, mode)
        self.keys = []

    def write_head(self, keys):
        assert len(self.keys) == 0, "keys are already defined"
        self.keys = keys
        self.f.write(','.join(keys) + '\n')

    def write_row(self, values):
        """row is a list of values matching the predifined keys"""
        assert len(values) == len(
            self.keys), "values are not consistent with predefined keys"
        s = ','.join(str(v) for v in values) + '\n'
        self.f.write(s)

    def write_df(self, df: DataFrame, progress=True):
        """write a dataframe to file"""
        keys = df.keys()
        self.write_head(keys)
        loop = zip(*[df[k] for k in keys])
        if progress:
            loop = tqdm(loop, total=len(df))
        for values in loop:
            self.write_row(values)

    def write(self, data):
        """write a dictionary"""
        if len(self.keys) == 0:
            self.write_head(data.keys())
        else:
            new_keys = data.keys() - self.keys

            if len(new_keys) > 0:
                # close the file
                self.f.close()
                # read the whole file as dataframe
                df = pd.read_csv(self.path)
                # add column in dataframe
                for k in new_keys:
                    df[k] = ''
                # clear keys
                self.keys = []
                # open the file again write mode
                self.f = open(self.path, 'w')
                # write the data frame to the file
                self.write_df(df, progress=False)

        # support skipping some keys
        values = []
        for k in self.keys:
            v = data.get(k)
            # missing values from the dict are written as blanks
            if v is None:
                v = ''
            values.append(v)
        self.write_row(values)
        self.f.flush()

    def close(self):
        self.f.close()
