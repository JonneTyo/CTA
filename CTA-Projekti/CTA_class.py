import numpy as np
import pandas as pd
from feature_data import *
import CTA_maths
import datetime



class CTAClass:

    def __init__(self, CTA_data_file_path):
        self.orig_path = CTA_data_file_path
        self.raw_data = csv_read(self.orig_path)

    @staticmethod
    def csv_read(file_path, index_col=0):
        import codecs
        def auto_truncate(val):
            return val[:255]

        s = ""
        with codecs.open(file_path, 'r+', encoding='utf-8', errors='ignore') as fdata:
            s = fdata.read()
            with open('Processed_data.csv', 'w', encoding='utf-8') as f:
                f.write(s)
        return pd.read_csv('Processed_data.csv', encoding='utf-8', converters={'label': auto_truncate}, index_col=index_col)

    def to_numpy(self, **kwargs):
        return self.raw_data.to_numpy(**kwargs)

    def str_to_num(self):
        pass

# for testing purposes
if __name__ == "__main__":
    cta_data = CTAClass(CURRENT_DATA_FILE)
    data_types = cta_data.raw_data.loc['Turku ID', :].dtypes
    print(data_types)



