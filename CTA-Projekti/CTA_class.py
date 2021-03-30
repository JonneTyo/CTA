import numpy as np
import pandas as pd
import feature_data
import CTA_maths
import datetime



class CTAClass:

    MAX_CATEGORICAL_VALUE = 20

    def __init__(self, CTA_data_file_path):
        self.orig_path = CTA_data_file_path
        self.raw_data = CTAClass.csv_read(self.orig_path)
        self.data = CTAClass.convert_dtypes(self.raw_data)
        self.handle_special_cases(**feature_data.CUSTOM_HANDLING)

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
        return self.data.to_numpy(**kwargs)

    # implements the changes to the data specified in CUSTOM_HANDLING
    def handle_special_cases(self, **kwargs):
        df = self.data
        for label, funcs in kwargs.items():
            for func_name, func_args in funcs.items():
                func = getattr(CTAClass, f'handle_{func_name}')
                df = func(df, label, **func_args)
        self.data = df
        pass

    @staticmethod
    def handle_missing(df, label=None, fill_val=0):
        if label is None:
            return df.fillna(value=fill_val)
        else:
            df.loc[:, label] = df.loc[:, label].fillna(value=fill_val)
            return df

    @staticmethod
    def handle_require(df, label, one_of=None, min_val=None, max_val=None):
        orig_filter = None
        if label == 'index':
            orig_filter = getattr(df, label)
        else:
            orig_filter = df.loc[:, label]
        filter = orig_filter.isin(orig_filter)
        if one_of is not None:
            filter = filter & orig_filter.isin(one_of)
        if min_val is not None:
            filter = filter & (orig_filter >= min_val)
        if max_val is not None:
            filter = filter & (orig_filter <= max_val)
        df = df.loc[filter, :]
        return df

    @staticmethod
    def handle_transform(df, label, combine, missing=np.nan):
        n_max = len(combine)
        for i, vals in enumerate(combine):
            df.loc[df.loc[:, label].isin(vals)] = i if i != missing else n_max
        return df

# for testing purposes
if __name__ == "__main__":
    cta_data = CTAClass(feature_data.CURRENT_DATA_FILE)
    data_types = cta_data.data.at[2, 'BMI']
    print(type(data_types))



