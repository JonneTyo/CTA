import numpy as np
import pandas as pd
import feature_data
import CTA_maths
import datetime



class CTAClass:
    MAX_CATEGORICAL_VALUE = 9
    def __init__(self, CTA_data_file_path):
        self.orig_path = CTA_data_file_path
        self.raw_data = CTAClass.csv_read(self.orig_path)
        self.data = CTAClass.convert_dtypes(self.raw_data)

    # returns a dataframe where each column has been converted to the best possible datatype
    @staticmethod
    def convert_dtypes(df):
        to_return = df.copy()
        for label, column in to_return.items():

            codes, uniques = column.factorize()

            # check if label is categorical
            if len(uniques) <= CTAClass.MAX_CATEGORICAL_VALUE + 1:
                to_return.loc[:, label] = codes
                continue

            # check if label is a date
            if 'date' in str(label).lower():
                try:
                    to_return.loc[:, label] = pd.to_datetime(column, infer_datetime_format=True)
                    continue
                except:
                    pass

            # attempt to convert the data to best suitable datatype. If that datatype is an object or a string, drop the label
            temp = column.convert_dtypes()
            temp_type = str(temp.dtype)
            if temp_type not in ('object', 'string'):
                to_return.loc[:, label] = temp
            else:
                to_return.drop(labels=label, axis=1, inplace=True)
        return to_return




    @staticmethod
    def csv_read(file_path, index_col=0):
        import codecs
        def remove_N(val):
            if val[0] == 'N':
                return int(val[1:])
            else:
                return int(val)

        s = ""
        with codecs.open(file_path, 'r+', encoding='utf-8', errors='ignore') as fdata:
            s = fdata.read()
            with open('Processed_data.csv', 'w', encoding='utf-8') as f:
                f.write(s)
        return pd.read_csv('Processed_data.csv', encoding='utf-8', converters={0: remove_N}, index_col=0)

    def to_numpy(self, **kwargs):
        return self.raw_data.to_numpy(**kwargs)

    def str_to_num(self):
        pass

# for testing purposes
if __name__ == "__main__":
    cta_data = CTAClass(feature_data.CURRENT_DATA_FILE)
    data_types = cta_data.data.at[2, 'BMI']
    print(type(data_types))



