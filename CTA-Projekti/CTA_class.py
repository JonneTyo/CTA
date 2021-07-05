import numpy as np
import pandas as pd
import CTA_maths
import datetime
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import RepeatedStratifiedKFold
import concurrent.futures
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

DATA_DIR = os.getcwd() + '\\Data'
ORIGINAL_DATA_DIR = DATA_DIR + '\\Original Data'
PROCESSED_DATA_DIR = DATA_DIR + '\\Processed Data'
RESULTS_DIR = os.getcwd() + '\\Results'
RESULTS_PET_DIR = os.getcwd() + '\\Results only pet'
COEFS_DIR = os.getcwd() + '\\Coefs'
COEFS_PET_DIR = os.getcwd() + '\\Coefs only pet'
PLOTS_DIR = os.getcwd() + '\\Plots'


# decorator which prints the change in the number of rows and columns the given function does
def print_change_in_patients(f):
    def wrapper_func(cta_class_obj, *args, **kwargs):
        temp = cta_class_obj.data
        rows, columns = temp.shape
        rv = f(cta_class_obj, *args, **kwargs)
        rows_new, columns_new = cta_class_obj.data.shape
        print(f'Function {f.__name__} caused a change of {rows_new - rows} patients and {columns_new - columns} variables \n')
        print(f'Current shape: {(rows_new, columns_new)}')
        return rv

    return wrapper_func

# extends the parameter the_list by the columns that were added by the function this function is decorating
def add_new_var_to_type_list(the_list):
    def decorator_func(f):
        def wrapper_func(cta_class_obj, *args, **kwargs):
            old_cols = cta_class_obj.data.columns
            rv = f(cta_class_obj, *args, **kwargs)
            new_cols = cta_class_obj.data.columns
            new_cols = new_cols.difference(old_cols)
            the_list.extend(new_cols)
            return rv
        return wrapper_func
    return decorator_func


class CTA_data_dictionary:
    dic_name = 'CTArekisteri_DataDictionary_2021-04-06.csv'
    dic_path = ORIGINAL_DATA_DIR + '\\' + dic_name

    def __init__(self):
        self.data = pd.read_csv(self.dic_path, header=0)
        self.data.at[317, 'Field Label'] = 'Tutkimus'
        self.data['Field Label Lower'] = [n.lower() for n in self.data['Field Label']]

    @property
    def cta(self):
        return self.get_field_labels('cta')

    @property
    def pet(self):
        return self.get_field_labels('perfuusio')

    @property
    def perustiedot(self):
        return self.get_field_labels('perustiedot')

    def get_field_labels(self, param, lower=True):
        if lower:
            return self.data.loc[self.data['Form Name'] == param]['Field Label Lower']
        return self.data.loc[self.data['Form Name'] == param]['Field Label']
        pass


class CTA_data_formatter:
    START_DATES = ['cta date']
    END_DATES = ['exitus date', 'date of death']
    END_OF_FOLLOW_UP_DATE = pd.Timestamp(datetime.date(2020, 12, 31))
    MAX_CATEGORICAL_VALUE = 20
    N_STENOSIS_TYPES = 3
    BASIC_VARIABLES = ['sex', 'age', 'passed time', 'bmi']
    RISK_VARIABLES = ['diabetes', 'smoking', 'chestpain', 'hypertension', 'dyslipidemia', 'dyspnea']
    CTA_VARIABLES = ['lm - type of stenosis', 'lada - type of stenosis', 'ladb - type of stenosis',
                     'ladc - type of stenosis', 'd1 - type of stenosis', 'd2 - type of stenosis',
                     'lcxa - type of stenosis', 'lcxb - type of stenosis', 'lcxc -type of stenosis',
                     'lpd - type of stenosis', 'lom1 - type of stenosis', 'lom2 - type of stenosis',
                     'im - type of stenosis', 'lpl - type of stenosis', 'rcaa - type of stenosis',
                     'rcab - type of stenosis', 'rcac - type of stenosis', 'rpd - type of stenosis',
                     'rpl - type of stenosis']
    PET_VARIABLES = [f'str_seg_{n}' for n in range(1, 18) if n not in [2, 3]]
    PET_SIGNIFICANCE_LEVEL = 2.3

    LABELS = ['mi or uap or all-cause death - status (2020)']
    CARDIO_LABELS = ['mi - confirmation', 'uap - confirmation', 'cardiovascular death - status (2020)']
    TIME_RESTRICTIONS = [n*365 for n in range(1, 9)]

    REQ_DIRS = [DATA_DIR, RESULTS_DIR, PLOTS_DIR, ORIGINAL_DATA_DIR, PROCESSED_DATA_DIR]
    CUSTOM_HANDLING = {
        # 'index': {
        #     'require': {'min_val': 862}
        # },
        'diabetes': {
            'transform': {'combine': [[3, 4, 5], [2, 1]],
                          'missing': 0},
        },
        # 'study indication': {
        #     'require': {'one_of': [1, 2, 4, 5, 11]}
        # },
        'chestpain': {
            'transform': {'combine': [[3,4], [2, 5], [1]],
                          'missing': 0}
        },
        'smoking': {
            'transform': {'combine': [[2], [1, 3]],
                          'missing': 0}
        }
    }

    def __init__(self, CTA_data_file_path=None, custom_variables=None):
        self.orig_path = CTA_data_file_path if CTA_data_file_path else CTA_data_formatter.get_data_paths()
        self.label_path = self.orig_path[0:len(ORIGINAL_DATA_DIR) + 19] + 'LABELS_' + self.orig_path[len(ORIGINAL_DATA_DIR) + 19:]
        self.raw_data = CTA_data_formatter.csv_read(self.orig_path)
        # self.b_combine_labels = b_combine_labels
        self.data = self.raw_data.copy()
        self.data.columns = CTA_data_formatter.csv_read(self.label_path).columns
        self.convert_dtypes()
        self.reinstate_nans()
        self.data.columns = self.columns
        self.handle_special_cases(**CTA_data_formatter.CUSTOM_HANDLING)
        self.drop_rows()
        #self.flip_binaries(['mi or uap or all-cause death - status (2020)'])
        self.pet_preprocessing()
        self.cta_preprocessing()
        if custom_variables:
            for key, method in custom_variables.items():
                method(self)
        #self.drop_patients_by_time()
        self.create_time_restricted_labels()
        self.drop_columns()
        self.remove_bad_patients()
        #self.fill_nas()

    @staticmethod
    def get_data_paths():
        return_path = ORIGINAL_DATA_DIR
        all_files = sorted(
            [f for f in os.listdir(return_path) if (os.path.isfile(os.path.join(return_path, f)) and ("CTArekisteri_DATA" in f and 'LABELS' not in f))])
        return return_path + '\\' + all_files[-1]

    @add_new_var_to_type_list(CTA_VARIABLES)
    @print_change_in_patients
    def calculate_min_cta_type(self):
        import re

        def min_above_a(x, a=-1):
            return x.loc[x > a].min()
        vars_to_use = [n for n in CTA_data_formatter.CTA_VARIABLES if re.search(" - type of stenosis$", n)]
        results = self.data.loc[:, vars_to_use].apply(min_above_a, axis=1)
        self.data['min cta type'] = results
        pass

    @add_new_var_to_type_list(CTA_VARIABLES)
    @print_change_in_patients
    def calculate_max_cta_type(self):
        import re

        vars_to_use = [n for n in CTA_data_formatter.CTA_VARIABLES if re.search(" - type of stenosis$", n)]
        results = self.data.loc[:, vars_to_use].max(axis=1)
        self.data['max cta type'] = results
        pass

    @add_new_var_to_type_list(CTA_VARIABLES)
    @print_change_in_patients
    def calculate_sis_values(self, variable_to_calculate="sis"):
        variable_th = 1
        if variable_to_calculate == "sss":
            variable_th = 2

        df = self.data.loc[:, self.CTA_VARIABLES]

        if 'stenosis type 1 count' in df.columns:
            df = df.loc[:, [f'stenosis type {n} count' for n in range(variable_th , self.N_STENOSIS_TYPES)]]
            self.data[variable_to_calculate] = df.apply(np.sum, axis=1)
        else:

            def count_freq(row):
                return len(row.loc[row >= variable_th])

            vals = df.apply(count_freq, axis=1)
            self.data[variable_to_calculate] = vals
        pass


    # returns a dataframe where each column has been converted to the best possible datatype
    @print_change_in_patients
    def convert_dtypes(self):
        to_return = self.data.copy()
        for label, column in to_return.items():

            # codes, uniques = column.factorize()
            #
            # # check if label is categorical
            # if len(uniques) <= CTA_data_formatter.MAX_CATEGORICAL_VALUE + 1:
            #     to_return.loc[:, label] = codes
            #     continue

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
                temp.fillna(value=-1, inplace=True)
                to_return.loc[:, label] = temp
            else:
                to_return.drop(labels=label, axis=1, inplace=True)
        self.data = to_return
        pass

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

    def drop_rows(self):
        self.data = self.data.loc[self.data['ams collab suspected cad'] == 1]
        pass

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

    def handle_require(self, label, one_of=None, min_val=None, max_val=None, how='all'):

        def combine(a, b, how):
            if how == 'all':
                return a & b
            else:
                return a | b

        df = self.data
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
        self.data = df
        pass

    @staticmethod
    def handle_transform(df, label, combine, missing=np.nan):
        n_max = len(combine)
        for i, vals in enumerate(combine):
            df.loc[df.loc[:, label].isin(vals), label] = i if i != missing else n_max
        return df

    # @print_change_in_patients
    # def keep_pectus(self):
    #     pectus = CTA_data_formatter.csv_read(CTA_data_formatter.PECTUS_IDS)
    #     filter1 = self.data.loc[:, 'study indication'] == 11
    #     filter2 = self.data.index.isin(pectus)
    #     self.data = self.data.loc[~(filter1 & filter2)]
    #     pass

    # calculates the time difference from start dates to end dates in days and saves it to a variable "passed time"
    @print_change_in_patients
    def calculate_passed_time(self, from_dates=False):
        if from_dates:
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
        else:
            def min_above_a(x, a=-1):
                vals = x.loc[x > a]
                return vals.min()

            date_cols = [n for n in self.data.columns if 'follow-up time' in n]
            new_dates = self.data.loc[:, date_cols]
            new_dates = new_dates.apply(min_above_a, axis=1)
            self.data['passed time'] = new_dates
            self.data.drop(date_cols, axis=1, inplace=True)
        pass

    # for each patient sums up the occurrence of each type of stenosis and stores them into self.data
    @print_change_in_patients
    def count_stenosis_types(self, n_classes=N_STENOSIS_TYPES):

        new_c_names = [f'stenosis type {n + 1} count' for n in range(-1, n_classes - 1)]

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
                count_vals[new_c_names[i]] = stenosis_data.apply(lambda x: countif(x, i - 1), axis=1)
                self.data[new_c_names[i]] = count_vals[new_c_names[i]]
                self.CTA_VARIABLES.append(new_c_names[i])

        pass

    def get_cta_and_pet_variables(self):
        to_return_cta = self.CTA_VARIABLES
        to_return_pet = self.PET_VARIABLES
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
            if col in all_vars or col in CTA_data_formatter.LABELS + CTA_data_formatter.CARDIO_LABELS:
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
            'cta': self.CTA_VARIABLES,
            'pet': self.PET_VARIABLES,
            'basic': CTA_data_formatter.BASIC_VARIABLES,
            'risk': CTA_data_formatter.RISK_VARIABLES
        }
        return to_return

    @add_new_var_to_type_list(LABELS)
    @print_change_in_patients
    def combine_labels(self):

        df1 = self.data.loc[:, CTA_data_formatter.LABELS]
        new_df = df1.max(axis=1)
        self.data['event'] = new_df
        self.data.drop(labels=CTA_data_formatter.LABELS, axis=1, inplace=True)
        self.data.drop(labels=CTA_data_formatter.CARDIO_LABELS, axis=1, inplace=True)
        self.LABELS.clear()
        pass

    @add_new_var_to_type_list(CARDIO_LABELS)
    @print_change_in_patients
    def combine_cv_labels(self):
        df2 = self.data.loc[:, CTA_data_formatter.CARDIO_LABELS]
        new_df = df2.max(axis=1)
        self.data['cv event'] = new_df
        self.data.drop(labels=CTA_data_formatter.CARDIO_LABELS, axis=1, inplace=True)
        pass

    @add_new_var_to_type_list(LABELS)
    @print_change_in_patients
    def create_time_restricted_labels(self, time_ths=TIME_RESTRICTIONS):

        for th in time_ths:
            new_vals = ((self.data['event'] == 1) & ((0 <= self.data['passed time']) & (self.data['passed time'] <= th))).astype(int)
            self.data[f'event {int(th/365)} years'] = new_vals
        pass

    @add_new_var_to_type_list(CARDIO_LABELS)
    @print_change_in_patients
    def create_time_restricted_cv_labels(self, time_ths=TIME_RESTRICTIONS):
        for th in time_ths:
            new_vals = ((self.data['cv event'] == 1) & ((0 <= self.data['passed time']) & (self.data['passed time'] <= th))).astype(int)
            self.data[f'cv event {th/365} years'] = new_vals
        pass

    @add_new_var_to_type_list(PET_VARIABLES)
    @print_change_in_patients
    def create_min_str_seg(self):
        str_segs = ['str_seg_' + str(n) for n in range(1, 18) if n not in [2,3]]
        self.data['min_str_seg'] = self.data.loc[:, str_segs].min(axis=1).apply(lambda x: min(10, max(0, x)))
        pass

    @add_new_var_to_type_list(PET_VARIABLES)
    @print_change_in_patients
    def create_max_str_seg(self):
        str_segs = ['str_seg_' + str(n) for n in range(1, 18) if n not in [2, 3]]
        self.data['max_str_seg'] = self.data.loc[:, str_segs].max(axis=1).apply(lambda x: min(5, max(0, x)))
        pass


    @print_change_in_patients
    def remove_bad_patients(self, ratio_breakpoint=2 / 3):
        variables_to_inspect = list()
        variables_to_inspect.extend(CTA_data_formatter.BASIC_VARIABLES)
        variables_to_inspect.extend(CTA_data_formatter.RISK_VARIABLES)
        variables_to_inspect.extend(self.CTA_VARIABLES)

        for val in range(1, CTA_data_formatter.N_STENOSIS_TYPES):
            variables_to_inspect.remove(f'stenosis type {val} count')
        variables_to_inspect.remove('passed time')

        n_variables = len(variables_to_inspect)
        to_drop = list()
        for index, row in self.data.loc[:, variables_to_inspect].iterrows():
            n_na = abs(row[row == -1].sum()) + abs(row.isna().sum())
            if n_na >= n_variables * ratio_breakpoint or self.raw_data.at[index, 'epaonn1___1'] == 1:
                to_drop.append(index)
        self.data.drop(labels=to_drop, axis=0, inplace=True)
        pass

    def fill_nas(self, fill_val=-1):
        self.data.fillna(value=fill_val, inplace=True)
        pass

    def flip_binaries(self, labels):
        def flip_non_nas(x):
            if x == 0 or x == 1:
                return abs(x - 1)
            return -1

        self.data.loc[:, labels] = self.data.loc[:, labels].applymap(flip_non_nas).astype(int)
        pass

    def reinstate_nans(self):
        temp = self.data.select_dtypes(include=['int64', 'float64'])
        temp = temp.applymap(lambda x: np.nan if (x == -1 or x == -1.0) else x)
        self.data.loc[:, temp.columns] = temp
        pass

    def pet_preprocessing(self):
        def relu(x):
            return np.max([0.0, self.PET_SIGNIFICANCE_LEVEL-x])
        df = self.data.loc[:, self.PET_VARIABLES]
        df = df.applymap(relu)
        self.data.loc[:, self.PET_VARIABLES] = df
        pass

    def cta_preprocessing(self):
        combine_dict = {
                        1: 0,
                        2: 1,
                        3: 1,
                        4: 2,
                        5: 2,
                        6: 2
                        }
        df = self.data.loc[:, self.CTA_VARIABLES]
        df = df.applymap(lambda x: combine_dict[x] if not np.isnan(x) else 0)
        self.data.loc[:, self.CTA_VARIABLES] = df
        pass

    @print_change_in_patients
    def drop_patients_by_time(self, time_max=5*365):
        df = self.data
        self.data = df.loc[df['passed time'] <= time_max]
        pass


class CTA_class:
    N_ITERATIONS = 100
    MODELS = ['lasso', 'SVC', 'linreg', 'SVR']
    METRICS = {'test_auc': 'Test AUC', 'sens': 'Sensitivity', 'spec': 'Specificity', 'accuracy': 'Accuracy'}
    BASIC_VARIABLES = ['sex', 'age', 'passed time', 'bmi']
    RISK_VARIABLES = ['diabetes', 'smoking', 'chestpain', 'hypertension', 'dyslipidemia', 'dyspnea']
    CTA_VARIABLES = ['stenosis type 1 count', 'stenosis type 2 count',
                     'lm - type of stenosis', 'lada - type of stenosis',
                     'ladb - type of stenosis', 'ladc - type of stenosis',
                     'd1 - type of stenosis', 'd2 - type of stenosis',
                     'lcxa - type of stenosis', 'lcxb - type of stenosis',
                     'lcxc -type of stenosis', 'lpd - type of stenosis',
                     'lom1 - type of stenosis', 'lom2 - type of stenosis',
                     'im - type of stenosis', 'lpl - type of stenosis',
                     'rcaa - type of stenosis', 'rcab - type of stenosis',
                     'rcac - type of stenosis', 'rpd - type of stenosis',
                     'rpl - type of stenosis', 'min cta type', 'max cta type', 'sis']
    PET_VARIABLES = ['str_seg_1', 'str_seg_4',
                    'str_seg_5', 'str_seg_6', 'str_seg_7', 'str_seg_8',
                    'str_seg_9', 'str_seg_10', 'str_seg_11', 'str_seg_12',
                   'str_seg_13', 'str_seg_14', 'str_seg_15', 'str_seg_16',
                    'str_seg_17', 'min_str_seg', 'max_str_seg']

    def __init__(self, label='event', all_labels=None, time=None, time_th=None, **settings):
        self.original_data = pd.read_csv(PROCESSED_DATA_DIR + '\\Processed data.csv', index_col=0, header=0)
        self.original_data = self.original_data.convert_dtypes()
        self.__dict__.update(settings)
        self.settings = settings
        self.label = label + f' {time} years' if time else label
        self.models = {i: j for (i, j) in zip(CTA_class.MODELS, [getattr(CTA_maths, f'maths_{n}_model')() for n in CTA_class.MODELS])}
        self.model_predictions = dict()
        self.keras_model = CTA_maths.maths_keras_model()
        self.keras_th = None
        self.time = time
        self.time_th = time_th
        self.label_time_th = label + f' {time_th} years' if time_th else self.label
        self.all_labels = all_labels

    def __call__(self, write_to_csv=True, n_splits=4, n_repeats=25):

        df = pd.DataFrame()
        coefs = pd.Series(index=self.data.columns, dtype='float64')
        n_iterations = n_splits*n_repeats
        train_test_splits = self.cta_train_test_split(n_splits=n_splits, n_iterations=n_repeats)
        for n, (X_train, X_test, y_train, y_test, yth_train, yth_test) in enumerate(train_test_splits):
            print(f'Iteration {n + 1} out of {n_iterations}')
            self.X_train, self.X_test, self.y_train, self.y_test, self.yth_train, self.yth_test = X_train, X_test, y_train, y_test, yth_train, yth_test
            self.train_models()
            self.get_coefs()
            self.predict(self.X_test)
            if df.shape[0] == 0:
                df = self.results
                coefs = self.coefs
            else:
                df = df + self.results
                coefs = coefs + self.coefs

        df = df / n_iterations
        coefs = coefs / n_iterations
        if write_to_csv:
            if hasattr(self, 'only_pet') and self.only_pet:
                df.to_csv(RESULTS_PET_DIR + f'\\results{self.settings_to_str}.csv')
                coefs.to_csv(COEFS_PET_DIR + f'\\coefs{self.settings_to_str}.csv')
            else:
                df.to_csv(RESULTS_DIR + f'\\results{self.settings_to_str}.csv')
                coefs.to_csv(COEFS_DIR + f'\\coefs{self.settings_to_str}.csv')
        return df

    @property
    def data(self):
        df = self.original_data.copy()
        time_dropped = False
        if hasattr(self, 'only_pet') and self.only_pet:
            df.dropna(how='all', subset=self.PET_VARIABLES, inplace=True)
        if hasattr(self, 'keep_basic') and not self.keep_basic:
            df.drop(labels=self.BASIC_VARIABLES + self.RISK_VARIABLES, axis=1, inplace=True)
            time_dropped = True
        if not self.keep_cta:
            df.drop(labels=self.CTA_VARIABLES, axis=1, inplace=True)
        if not self.keep_pet:
            df.drop(labels=self.PET_VARIABLES, axis=1, inplace=True)

        df.drop(index=df.loc[df[self.label] == -1.0].index, inplace=True)
        df.dropna(axis=0, how='any', subset=['event'], inplace=True)
        df.dropna(axis=0, how='any', subset=[self.label], inplace=True)
        if not time_dropped:
            df.drop(columns='passed time', inplace=True)
        df.drop(labels=[n for n in df.columns if 'event' in n], axis=1, inplace=True)
        return df

    @property
    def data_fill_na(self):
        df = self.data
        fill_dict = {n: df[n].median() for n in df.columns}
        for label in self.CTA_VARIABLES:
            if 'type of stenosis' in label:
                fill_dict[label] = 0
        for label in self.PET_VARIABLES:
            if 'str_seg_' in label:
                fill_dict[label] = 0
        df.fillna(fill_dict, inplace=True)
        return df

    @property
    def labels(self):
        return self.original_data.loc[self.data.index, self.label]

    @property
    def labels_time_th(self):
        return self.original_data.loc[self.data.index, self.label_time_th]

    @property
    def metric_names(self):
        return [n for n in CTA_class.METRICS.values()]

    @property
    def settings_to_str(self):
        to_return = ''
        for key, val in self.settings.items():
            if val:
                to_return += '_' + key
        if self.time:
            to_return += f'_{self.time}_years'
        if self.label_time_th != self.label:
            to_return += f'_inspect_{self.time_th}_years'
        return to_return

    @property
    def keras_weights_file(self):
        return 'keras_' + self.settings_to_str

    def to_numpy(self):
        return self.data.to_numpy()

    # splits the data into training and testing portions so that the ratio of positive events stays the same in both
    def cta_train_test_split(self, n_splits=4, n_iterations=25, random_state=None):
        data = self.data_fill_na
        labels = self.labels.astype(int)
        labels_time_th = self.labels_time_th.astype(int)
        rskf = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_iterations, random_state=random_state)
        for train_ind, test_ind in rskf.split(data, labels):
            X_train, X_test = data.iloc[train_ind, :], data.iloc[test_ind, :]
            y_train, y_test = labels.iloc[train_ind], labels.iloc[test_ind]
            yth_train, yth_test = labels_time_th.iloc[train_ind], labels_time_th.iloc[test_ind]
            yield X_train, X_test, y_train, y_test, yth_train, yth_test


        # from sklearn.model_selection import train_test_split
        # if random_state is None:
        #     random_state = datetime.datetime.now().microsecond % 1000
        # pos_filter, neg_filter = labels > 0, labels <= 0
        # pos_d, pos_l, neg_d, neg_l = data[pos_filter], labels[pos_filter], data[neg_filter], labels[neg_filter]
        #
        # x_train1, x_test1, y_train1, y_test1 = train_test_split(pos_d, pos_l, train_size=train_ratio, random_state=random_state)
        # x_train2, x_test2, y_train2, y_test2 = train_test_split(neg_d, neg_l, train_size=train_ratio, random_state=random_state)
        # x_train1 = np.append(x_train1, x_train2, axis=0)
        # x_test1 = np.append(x_test1, x_test2, axis=0)
        # y_train1 = np.append(y_train1.to_frame(), y_train2.to_frame(), axis=0)
        # y_test1 = np.append(y_test1.to_frame(), y_test2.to_frame(), axis=0)
        # x_train1 = np.append(x_train1, y_train1, axis=1)
        # x_test1 = np.append(x_test1, y_test1, axis=1)
        # np.random.shuffle(x_train1)
        # np.random.shuffle(x_test1)
        # x_train1 = x_train1.astype(float)
        # x_test1 = x_test1.astype(float)
        #
        # return x_train1[:, :-y_train1.shape[1]], x_test1[:, :-y_test1.shape[1]], x_train1[:, -y_train1.shape[1]:].ravel(), x_test1[:,
        #                                                                                                                    -y_test1.shape[
        #                                                                                                                        1]:].ravel()

    def train_models(self):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {model_name: executor.submit(model.fit, self.X_train, self.y_train) for (model_name, model) in self.models.items()}
        pass

    def predict(self, X):
        for model_name, model in self.models.items():
            self.model_predictions[model_name] = model.predict(X)
        pass

    @property
    def results(self):

        result_df = pd.DataFrame(index=CTA_class.MODELS, columns=self.metric_names)
        for model_name, pred in self.model_predictions.items():
            for metric, _ in CTA_class.METRICS.items():
                metric_f = getattr(self, metric)
                result_df.at[model_name, CTA_class.METRICS[metric]] = metric_f(pred)
        return result_df

    def test_auc(self, given_pred):
        return roc_auc_score(self.yth_test, given_pred)

    def sens(self, given_pred):

        def youden(sens, spec):
            return sens[np.argmax(sens + spec)]

        fpr, sens, _ = roc_curve(self.yth_test, given_pred)
        spec = 1 - fpr
        return youden(sens, spec)

    def spec(self, given_pred):

        def youden(sens, spec):
            return spec[np.argmax(sens + spec)]

        fpr, sens, _ = roc_curve(self.yth_test, given_pred)
        spec = 1 - fpr
        return youden(sens, spec)

    def accuracy(self, given_pred):

        def distance_from_one(val):
            return abs(1 - val)

        def youden(sens, spec, th):
            return th[np.argmax(sens + spec)]

        fpr, sens, ths = roc_curve(self.yth_test, given_pred)
        spec = 1 - fpr
        th = youden(sens, spec, ths)
        pred_classes = given_pred >= th
        vals = self.yth_test + pred_classes
        total = 0
        for v in map(distance_from_one, vals):
            total += v
        return total / len(given_pred)

    def train_keras(self, new_weights_save_path=os.getcwd() + '\\keras_weights.hdf5', old_weights_save_path=None, n_epochs=20):

        x_train = self.reshape_for_keras(self.X_train).astype('float64')
        y_train = self.y_train.astype('float64').to_numpy()
        x_test = self.reshape_for_keras(self.X_test).astype('float64')
        y_test = self.y_test.astype('float64').to_numpy()

        x_val = x_train[-int(x_train.shape[0]/5):]
        y_val = y_train[-int(y_train.shape[0]/5):]
        x_train = x_train[:-int(x_train.shape[0]/5)]
        y_train = y_train[:-int(y_train.shape[0]/5)]

        if old_weights_save_path:
            self.keras_model.load_weights(old_weights_save_path)

        model_checkpoint = keras.callbacks.ModelCheckpoint(new_weights_save_path,
                                                           monitor='loss', verbose=1,
                                                           save_best_only=True)
        history = self.keras_model.fit(
            x_train,
            y_train,
            batch_size=32,
            epochs=n_epochs,
            validation_data=(x_val, y_val),
            callbacks=[model_checkpoint]
        )
        print("Evaluate on test data")
        results = self.keras_model.evaluate(x_test,y_test, batch_size=128)
        print('Mean Squared Error, Accuracy, AUC, TruePositives, FalseNegatives, FalsePositives')
        print(results)

        self.keras_th = self.calculate_keras_th(x_val, y_val)
        pass

    def keras_predict(self):
        x = self.reshape_for_keras(self.X_test).astype('float64')
        pred = self.keras_model.predict(x).flatten()
        self.model_predictions['keras'] = pred
        pass

    # return th for predicting binary values
    def calculate_keras_th(self, x_val, y_val):

        def youden(sens, spec, th):
            return th[np.argmax(sens + spec)]

        pred = self.keras_model.predict(x_val)
        pred = pred.flatten()
        fpr, sens, ths = roc_curve(y_val,pred)
        spec = 1-fpr
        return youden(sens, spec, ths)

    def reshape_for_keras(self, X):
        n_padding = 17 - X.shape[1]
        n_padding = np.zeros((X.shape[0], n_padding))
        x_train = np.concatenate((X, n_padding), axis=1)
        return x_train

    def get_coefs(self):
        self.coefs = pd.Series(data=self.models['lasso'].coef_, index=self.data.columns)
        pass


# create required directories if they do not exist
for val in CTA_data_formatter.REQ_DIRS:
    try:
        os.mkdir(val)
        print('Created directory: ' + val)
    except OSError:
        pass


def delete_folder_contents(dir=RESULTS_DIR):
    import glob
    files = glob.glob(dir + '/*')
    for f in files:
        os.remove(f)
    pass

# iterate through all desired dataset settings
def iter_options(settings, n_start=0):
    n_options = 2 ** len(settings)
    for i in range(n_start, n_options):
        b_str = bin(i)[2:]

        # pad the beginning of b_str with zeroes if needed
        if len(b_str) < len(settings):
            pad = len(settings) - len(b_str)
            for j in range(pad):
                b_str = '0' + b_str

        for j, (key, item) in zip(b_str, settings.items()):
            settings[key] = bool(int(j))

        # skip settings possibility that pet are kept but cta are not
        #if (not settings['keep_cta']) and settings['keep_pet']:
        #    continue
        yield settings


CUSTOM_VARIABLES = {
        'passed time': CTA_data_formatter.calculate_passed_time,
        'stenosis type counts': CTA_data_formatter.count_stenosis_types,
        'events': CTA_data_formatter.combine_labels,
        #'cv events': CTA_data_formatter.combine_cv_labels,
        #'timed events': CTA_data_formatter.create_time_restricted_labels,
        #'timed cv events': CTA_data_formatter.create_time_restricted_cv_labels,
        'min cta type': CTA_data_formatter.calculate_min_cta_type,
        'max cta type': CTA_data_formatter.calculate_max_cta_type,
        'sis': CTA_data_formatter.calculate_sis_values,
        'min str seg': CTA_data_formatter.create_min_str_seg,
        'max str seg': CTA_data_formatter.create_max_str_seg,
    }

if __name__ == "__main__":
    format_original_data = False
    iterate_all_options = True
    iteration_start = 0
    train_keras = False
    train_models = True
    all_labels = None


    settings = {
        'only_pet': True,
        #'timed': False,
        'keep_cta': True,
        'keep_pet': True,
        #'cv': False
    }
    training_times = [n for n in range(6)]

    if format_original_data:
        cta_data = CTA_data_formatter(custom_variables=CUSTOM_VARIABLES)
        print((cta_data.data['event'] == 1).sum())
        cta_data.to_csv()

    if train_models:
        curr_iter = 1
        delete_folder_contents()
        delete_folder_contents(RESULTS_PET_DIR)
        for opt in iter_options(settings, iteration_start):
            for train_t in training_times:
                print(f'\n Starting settings no. {curr_iter} with {train_t} years.')
                cta_data = None
                if train_t == 0:
                    cta_data = CTA_class(all_labels=all_labels, **opt)
                else:
                    cta_data = CTA_class(all_labels=all_labels, time=train_t, **opt)
                cta_data()

            curr_iter += 1

    settings = {
        'only_pet': False,
        'keep_basic': False,
        'keep_cta': False,
        'keep_pet': True,
        'cv': False
    }
    if train_keras:
        physical_devices = tf.config.list_physical_devices('GPU')
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
        results = {}
        for n in [5, 10, 20, 30, 40]:
            cta_data = CTA_class(time=4, **settings)
            train_test_splits = cta_data.cta_train_test_split(n_splits=4, n_iterations=1)
            results[n] = None
            for i, (X_train, X_test, y_train, y_test, yth_train, yth_test) in enumerate(train_test_splits):
                cta_data.keras_model = CTA_maths.maths_keras_model()
                cta_data.X_train, cta_data.X_test, cta_data.y_train, cta_data.y_test, cta_data.yth_train, cta_data.yth_test = X_train, X_test, y_train, y_test, yth_train, yth_test
                cta_data.train_keras(n_epochs=n)
                cta_data.keras_predict()
                if results[n] is None:
                    results[n] = cta_data.results.loc['keras', :]
                else:
                    results[n] = results[n] + cta_data.results.loc['keras', :]

            results[n] = results[n]/4
        for key, value in results.items():
            print(f'No. of Epochs: {key} \n {value}')
