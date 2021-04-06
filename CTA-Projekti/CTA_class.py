import numpy as np
import pandas as pd
import CTA_maths
import datetime
import os


# decorator which prints the change in the number of rows and columns the given function does
def print_change_in_patients(f):
    def wrapper_func(df, *args, **kwargs):
        if isinstance(df, (CTA_data_formatter, CTA_class)):
            temp = df.data
        else:
            temp = df
        rows, columns = temp.shape
        rv = f(df, *args, **kwargs)
        if isinstance(df, (CTA_data_formatter, CTA_class)):
            rows_new, columns_new = df.data.shape
        else:
            rows_new, columns_new = df.shape
        print(f'Function {f.__name__} caused a change of {rows_new - rows} patients and {columns_new - columns} variables \n')
        return rv
    return wrapper_func


class CTA_data_formatter:

    DATA_DIR = os.getcwd() + '\\Data'
    ORIGINAL_DATA_DIR = DATA_DIR + '\\Original Data'
    PROCESSED_DATA_DIR = DATA_DIR + '\\Processed Data'
    RESULTS_DIR = os.getcwd() + '\\Results'
    PLOTS_DIR = os.getcwd() + '\\Plots'
    CURRENT_DATA_FILE = ORIGINAL_DATA_DIR + '\\CTArekisteri_DATA_LABELS_2021-02-17_1146.csv'
    PECTUS_IDS = ORIGINAL_DATA_DIR + '\\PECTUS.csv'
    START_DATES = ['cta date']
    END_DATES = ['exitus date', 'date of death']
    END_OF_FOLLOW_UP_DATE = pd.Timestamp(datetime.date(2020, 12, 31))
    MAX_CATEGORICAL_VALUE = 20
    N_STENOSIS_TYPES = 7
    BASIC_VARIABLES = ['sex', 'age', 'passed time']
    RISK_VARIABLES = ['diabetes', 'smoking', 'chestpain', 'hypertension', 'dyslipidemia', 'dyspnea']
    CTA_VARIABLES, PET_VARIABLES = [f'stenosis type {n} count' for n in range(N_STENOSIS_TYPES)], []

    LABELS = ['uap1 - confirmed', 'mi1 - confirmed', 'date of death']
    TIME_RESTRICTION = 4*365

    REQ_DIRS = [DATA_DIR, RESULTS_DIR, PLOTS_DIR, ORIGINAL_DATA_DIR, PROCESSED_DATA_DIR]
    CUSTOM_HANDLING = {
        'index': {
            'require': {'min_val': 862}
        },
        'diabetes': {
            'missing': {'fill_val': 4},
            'transform': {'combine': [(1, 2)],
                          'missing': 0},
        },
        'study indication': {
            'require': {'one_of': [1, 2, 4, 5, 11]}
        },
        'chestpain': {
            'missing': {'fill_val': 3}
        },
        'smoking': {
            'transform': {'combine': [(1, 3)],
                          'missing': 0}
        }
    }

    def __init__(self, CTA_data_file_path, b_combine_labels=True):
        self.orig_path = CTA_data_file_path
        self.raw_data = CTA_data_formatter.csv_read(self.orig_path)
        self.b_combine_labels = b_combine_labels
        CTA_data_formatter.CTA_VARIABLES, CTA_data_formatter.PET_VARIABLES = self.get_cta_and_pet_variables()
        self.data = CTA_data_formatter.convert_dtypes(self.raw_data)
        self.data.columns = self.columns
        self.handle_special_cases(**CTA_data_formatter.CUSTOM_HANDLING)
        self.keep_pectus()
        self.calculate_passed_time()
        self.count_stenosis_types()
        self.drop_columns()
        if b_combine_labels:
            self.combine_labels()
        self.create_time_restricted_labels()

    # returns a dataframe where each column has been converted to the best possible datatype
    @staticmethod
    @print_change_in_patients
    def convert_dtypes(df):
        to_return = df.copy()
        for label, column in to_return.items():

            codes, uniques = column.factorize()

            # check if label is categorical
            if len(uniques) <= CTA_data_formatter.MAX_CATEGORICAL_VALUE + 1:
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
            temp = column.convert_dtypes(convert_integer=True)
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
        return pd.read_csv('Processed_data.csv', encoding='utf-8', converters={0: remove_N}, index_col=index_col)

    def to_csv(self, file_path=PROCESSED_DATA_DIR + '\\Processed data.csv'):
        self.data.to_csv(file_path)

    # implements the changes to the data specified in CUSTOM_HANDLING
    @print_change_in_patients
    def handle_special_cases(self, **kwargs):
        df = self.data
        for label, funcs in kwargs.items():
            for func_name, func_args in funcs.items():
                func = getattr(CTA_data_formatter, f'handle_{func_name}')
                df = func(df, label, **func_args)
        self.data = df
        pass

    @staticmethod
    def handle_missing(df, label=None, fill_val=0):
        if label is None:
            return df.fillna(value=fill_val)
        else:
            df.loc[:, label] = df.loc[:, label].fillna(value=fill_val)
            df.loc[:, label].loc[df[label] == -1] = fill_val
            return df

    @staticmethod
    def handle_require(df, label, one_of=None, min_val=None, max_val=None, how='all'):

        def combine(a, b, how):
            if how == 'all':
                return a & b
            else:
                return a | b

        orig_filter = None
        if label == 'index':
            orig_filter = getattr(df, label)
        else:
            orig_filter = df.loc[:, label]
        if how == 'all':
            filter = orig_filter.isin(orig_filter)
        else:
            filter = orig_filter.isin([])
        if one_of is not None:
            filter = combine(filter, orig_filter.isin(one_of), how)
        if min_val is not None:
            filter = combine(filter, (orig_filter >= min_val), how)
        if max_val is not None:
            filter = combine(filter, (orig_filter <= max_val), how)
        df = df.loc[filter, :]
        return df

    @staticmethod
    def handle_transform(df, label, combine, missing=np.nan):
        n_max = len(combine)
        for i, vals in enumerate(combine):
            df.loc[df.loc[:, label].isin(vals), label] = i if i != missing else n_max
        return df

    @print_change_in_patients
    def keep_pectus(self):
        pectus = CTA_data_formatter.csv_read(CTA_data_formatter.PECTUS_IDS)
        filter1 = self.data.loc[:, 'study indication'] == 11
        filter2 = self.data.index.isin(pectus)
        self.data = self.data.loc[~(filter1 & filter2)]
        pass

    # calculates the time difference from start dates to end dates in days and saves it to a variable "passed time"
    @print_change_in_patients
    def calculate_passed_time(self):

        # get the earliest date of start dates and latest date of end dates
        end_date = self.data.loc[:, CTA_data_formatter.END_DATES]
        if end_date.shape[1] > 1:
            end_date = end_date.max(axis=1)
        else:
            end_date = end_date.squeeze()
        start_date = self.data.loc[:, CTA_data_formatter.START_DATES]
        if start_date.shape[1] > 1:
            start_date = start_date.min(axis=1)
        else:
            start_date = start_date.squeeze()

        # drop indexes with start dates missing
        start_na = start_date.notna()
        end_date = end_date.loc[start_na]
        self.data = self.data.loc[start_na]
        start_date.dropna(inplace=True)

        # fill the missing values in end_date with a pre-determined value
        end_date.fillna(CTA_data_formatter.END_OF_FOLLOW_UP_DATE, inplace=True)

        # calculate values in days
        vals = (end_date - start_date).array.days

        self.data['passed time'] = vals
        pass

    # for each patient sums up the occurrence of each type of stenosis and stores them into self.data
    @print_change_in_patients
    def count_stenosis_types(self, n_classes=N_STENOSIS_TYPES):

        new_c_names = [f'stenosis type {n + 1} count' for n in range(-1, n_classes-1)]

        stenosis_data = self.data.loc[:, [n for n in self.data.columns if 'type of stenosis' in n.lower()]]

        def countif(values, i):
            count = 0
            if i != -1:
                for val in values:
                    if pd.isnull(val):
                        continue
                    if val == i:
                        count += 1
            else:
                for val in values:
                    if pd.isnull(val):
                        count += 1
            return count

        count_vals = dict()
        for i in range(n_classes):
            if i > 0:
                count_vals[new_c_names[i]] = stenosis_data.apply(lambda x: countif(x, i-1), axis=1)
                self.data[new_c_names[i]] = count_vals[new_c_names[i]]

        pass

    def get_cta_and_pet_variables(self):
        to_return_cta = list()
        to_return_pet = list()
        for col in self.raw_data.columns:
            if 'type of stenosis' in str(col).lower():
                to_return_cta.append(col.lower())
            elif 'str_seg' in str(col).lower():
                to_return_pet.append(col.lower())

        return to_return_cta, to_return_pet

    @print_change_in_patients
    def drop_columns(self):
        all_vars = []
        for key, item in self.all_variables.items():
            for i in item:
                all_vars.append(i.lower())
        for col in self.columns:
            if col in all_vars or col in CTA_data_formatter.LABELS:
                continue
            self.data.drop(labels=col, axis=1, inplace=True)
        pass

    @property
    def columns(self):
        for i in self.data.columns:
            yield i.lower()
        pass

    @property
    def original_columns(self):
        for i in self.raw_data.columns:
            yield i
        pass

    @property
    def all_variables(self):
        to_return = {
            'cta': CTA_data_formatter.CTA_VARIABLES,
            'pet': CTA_data_formatter.PET_VARIABLES,
            'basic': CTA_data_formatter.BASIC_VARIABLES,
            'risk': CTA_data_formatter.RISK_VARIABLES
        }
        return to_return

    def combine_labels(self):

        df = self.data.loc[:, CTA_data_formatter.LABELS]

        for i, dt in enumerate(df.dtypes):
            if 'int' not in str(dt).lower():
                df.iloc[:, i] = df.iloc[:, i].notna().astype(int)
        new_df = df.max(axis=1)
        self.data['event'] = new_df
        self.data.drop(labels=CTA_data_formatter.LABELS, axis=1, inplace=True)
        pass

    def create_time_restricted_labels(self, min_d=0, max_d=TIME_RESTRICTION):
        new_vals = list()
        for ind, row in self.data.iterrows():
            if row['event'] != 1:
                new_vals.append(0)
                continue
            if min_d <= row['passed time'] and row['passed time'] <= max_d:
                new_vals.append(1)
            else:
                new_vals.append(0)

        self.data['timed event'] = new_vals
        pass



class CTA_class:

    def __init__(self, combine_labels=True):
        self.original_data = pd.read_csv(CTA_data_formatter.PROCESSED_DATA_DIR + '\\Processed data.csv', index_col=0)
        self.original_data = self.original_data.convert_dtypes()
        self.combine_labels = combine_labels


# create required directories if they do not exist
for dir in CTA_data_formatter.REQ_DIRS:
    try:
        os.mkdir(dir)
        print('Created directory: ' + dir)
    except OSError:
        pass

# for testing purposes
if __name__ == "__main__":

    format_original_data = True

    if format_original_data:
        cta_data = CTA_data_formatter(CTA_data_formatter.CURRENT_DATA_FILE)
        cta_data.to_csv()

    cta_data = CTA_class()
    print(cta_data.original_data)


    # todo y_labels, new class for the x_labels
