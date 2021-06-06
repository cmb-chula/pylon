import os

from pandas import DataFrame
import pandas as pd
import csv

from tqdm.autonotebook import tqdm


class FastCSVWriter:
    def __init__(self, path, mode='w', quoting=csv.QUOTE_MINIMAL):
        dirname = os.path.dirname(path)
        if dirname != '' and not os.path.exists(dirname):
            os.makedirs(dirname)

        self.path = path
        self.f = open(path, mode)
        self.quoting = quoting
        self.writer = csv.writer(self.f, quoting=self.quoting)
        self.keys = []

    def write_head(self, keys):
        assert len(self.keys) == 0, "keys are already defined"
        self.keys = keys
        # self.f.write(','.join(keys) + '\n')
        self.writer.writerow(keys)

    def write_row(self, values):
        """row is a list of values matching the predifined keys"""
        assert len(values) == len(
            self.keys), "values are not consistent with predefined keys"
        # s = ','.join(str(v) for v in values) + '\n'
        # self.f.write(s)
        self.writer.writerow(values)

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
                self.writer = csv.writer(self.f, quoting=self.quoting)
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
