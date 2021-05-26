import numpy as np
import pandas as pd
import CTA_maths
import datetime
import os
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import concurrent.futures

DATA_DIR = os.getcwd() + '\\Data'
ORIGINAL_DATA_DIR = DATA_DIR + '\\Original Data'
PROCESSED_DATA_DIR = DATA_DIR + '\\Processed Data'
RESULTS_DIR = os.getcwd() + '\\Results'
PLOTS_DIR = os.getcwd() + '\\Plots'


# decorator which prints the change in the number of rows and columns the given function does
def print_change_in_patients(f):
    def wrapper_func(cta_class_obj, *args, **kwargs):
        temp = cta_class_obj.data
        rows, columns = temp.shape
        rv = f(cta_class_obj, *args, **kwargs)
        rows_new, columns_new = cta_class_obj.data.shape
        print(f'Function {f.__name__} caused a change of {rows_new - rows} patients and {columns_new - columns} variables \n')
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
    N_STENOSIS_TYPES = 7
    BASIC_VARIABLES = ['sex', 'age', 'passed time']
    RISK_VARIABLES = ['diabetes', 'smoking', 'chestpain', 'hypertension', 'dyslipidemia', 'dyspnea']
    CTA_VARIABLES = ['stenosis type 1 count', 'stenosis type 2 count', 'stenosis type 3 count',
                     'stenosis type 4 count', 'stenosis type 5 count', 'stenosis type 6 count',
                     'lm - type of stenosis', 'lada - type of stenosis', 'ladb - type of stenosis',
                     'ladc - type of stenosis', 'd1 - type of stenosis', 'd2 - type of stenosis',
                     'lcxa - type of stenosis', 'lcxb - type of stenosis', 'lcxc -type of stenosis',
                     'lpd - type of stenosis', 'lom1 - type of stenosis', 'lom2 - type of stenosis',
                     'im - type of stenosis', 'lpl - type of stenosis', 'rcaa - type of stenosis',
                     'rcab - type of stenosis', 'rcac - type of stenosis', 'rpd - type of stenosis',
                     'rpl - type of stenosis']
    PET_VARIABLES = [f'str_seg_{n}' for n in range(1, 18)]

    LABELS = ['mi or uap or all-cause death - status (2020)']
    CARDIO_LABELS = ['mi - confirmation', 'uap - confirmation', 'cardiovascular death - status (2020)']
    TIME_RESTRICTION = 6 * 365

    REQ_DIRS = [DATA_DIR, RESULTS_DIR, PLOTS_DIR, ORIGINAL_DATA_DIR, PROCESSED_DATA_DIR]
    CUSTOM_HANDLING = {
        # 'index': {
        #     'require': {'min_val': 862}
        # },
        'diabetes': {
            'missing': {'fill_val': 4},
            'transform': {'combine': [(1, 2)],
                          'missing': 0},
        },
        # 'study indication': {
        #     'require': {'one_of': [1, 2, 4, 5, 11]}
        # },
        'chestpain': {
            'missing': {'fill_val': 3}
        },
        'smoking': {
            'transform': {'combine': [(1, 3)],
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
        self.flip_binaries(['mi or uap or all-cause death - status (2020)'])
        if custom_variables:
            for key, method in custom_variables.items():
                method(self)
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

    # returns a dataframe where each column has been converted to the best possible datatype
    @print_change_in_patients
    def convert_dtypes(self):
        to_return = self.data.copy()
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
        self.data = self.data.loc[self.data['ams collab suspected cad'] == 0]
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
    def create_time_restricted_labels(self, min_d=0, max_d=TIME_RESTRICTION):
        new_vals = ((self.data['event'] == 1) & ((min_d <= self.data['passed time']) & (self.data['passed time'] <= max_d))).astype(int)
        self.data['timed event'] = new_vals
        pass

    @add_new_var_to_type_list(CARDIO_LABELS)
    @print_change_in_patients
    def create_time_restricted_cv_labels(self, min_d=0, max_d=TIME_RESTRICTION):
        new_vals = ((self.data['cv event'] == 1) & ((min_d <= self.data['passed time']) & (self.data['passed time'] <= max_d))).astype(int)
        self.data['timed cv event'] = new_vals
        pass

    @add_new_var_to_type_list(PET_VARIABLES)
    @print_change_in_patients
    def create_min_str_seg(self):
        str_segs = ['str_seg_' + str(n) for n in range(1, 18)]
        self.data['min_str_seg'] = self.data.loc[:, str_segs].min(axis=1).apply(lambda x: min(10, max(0, x)))
        pass

    @add_new_var_to_type_list(PET_VARIABLES)
    @print_change_in_patients
    def create_max_str_seg(self):
        str_segs = ['str_seg_' + str(n) for n in range(1, 18)]
        self.data['max_str_seg'] = self.data.loc[:, str_segs].max(axis=1).apply(lambda x: min(10, max(0, x)))
        pass


    @print_change_in_patients
    def remove_bad_patients(self, ratio_breakpoint=2 / 3):
        variables_to_inspect = CTA_data_formatter.BASIC_VARIABLES
        variables_to_inspect.extend(CTA_data_formatter.RISK_VARIABLES)
        variables_to_inspect.extend(self.CTA_VARIABLES)

        for val in range(1, CTA_data_formatter.N_STENOSIS_TYPES):
            variables_to_inspect.remove(f'stenosis type {val} count')
        variables_to_inspect.remove('passed time')

        n_variables = len(variables_to_inspect)
        to_drop = list()
        for index, row in self.data.loc[:, variables_to_inspect].iterrows():
            n_na = abs(row[row == -1].sum())
            if n_na >= n_variables * ratio_breakpoint:
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


class CTA_class:
    N_ITERATIONS = 100
    MODELS = ['lasso', 'SVC', 'linreg']
    METRICS = {'test_auc': 'Test AUC', 'sens': 'Sensitivity', 'spec': 'Specificity', 'accuracy': 'Accuracy'}

    def __init__(self, label='event', **settings):
        self.original_data = pd.read_csv(PROCESSED_DATA_DIR + '\\Processed data.csv', index_col=0, header=0)
        self.original_data = self.original_data.convert_dtypes()
        self.__dict__.update(settings)
        self.settings = settings
        self.label = int(self.settings['timed'])*'timed ' + int(self.settings['cv'])*'cv ' + label
        self.models = {i: j for (i, j) in zip(CTA_class.MODELS, [getattr(CTA_maths, f'maths_{n}_model')() for n in CTA_class.MODELS])}
        self.X_train, self.X_test, self.y_train, self.y_test = self.cta_train_test_split(self.labels)
        self.model_predictions = dict()

    def __call__(self, write_to_csv=True):

        df = pd.DataFrame()
        for n in range(CTA_class.N_ITERATIONS):
            print(f'Iteration {n} out of {CTA_class.N_ITERATIONS}')
            self.X_train, self.X_test, self.y_train, self.y_test = self.cta_train_test_split(self.labels)
            self.train_models()
            self.predict(self.X_test)
            if df.shape[0] == 0:
                df = self.results
            else:
                df = df + self.results

        df = df / CTA_class.N_ITERATIONS
        if write_to_csv:
            df.to_csv(RESULTS_DIR + f'\\results_{self.settings_to_str}.csv')
        return df

    @property
    def data(self):
        df = self.original_data.copy()
        if self.only_pet:
            df.dropna(how='all', subset=[f'str_seg_{n + 1}' for n in range(17)], inplace=True)
        if not self.keep_cta:
            df.drop(labels=CTA_data_formatter.CTA_VARIABLES, axis=1, inplace=True)
        if not self.keep_pet:
            df.drop(labels=CTA_data_formatter.PET_VARIABLES, axis=1, inplace=True)

        df.drop(index=df.loc[df[self.label] == -1.0].index, inplace=True)
        df.drop(columns='passed time', inplace=True)
        df.drop(labels=['event', 'timed event', 'cv event', 'timed cv event'], axis=1, inplace=True)
        return df

    @property
    def data_fill_na(self):
        df = self.data
        fill_dict = {n: df[n].median() for n in df.columns}
        df.fillna(fill_dict, inplace=True)
        return df


    @property
    def labels(self):
        return self.original_data.loc[self.data.index, self.label]

    @property
    def metric_names(self):
        return [n for n in CTA_class.METRICS.values()]

    @property
    def settings_to_str(self):
        to_return = ''
        for key, val in self.settings.items():
            if val:
                to_return += '_' + key
        return to_return

    def to_numpy(self):
        return self.data.to_numpy()

    # splits the data into training and testing portions so that the ratio of positive events stays the same in both
    def cta_train_test_split(self, labels, train_ratio=3 / 4, random_state=None):
        data = self.data_fill_na
        from sklearn.model_selection import train_test_split
        if random_state is None:
            random_state = datetime.datetime.now().microsecond % 1000
        pos_filter, neg_filter = labels > 0, labels <= 0
        pos_d, pos_l, neg_d, neg_l = data[pos_filter], labels[pos_filter], data[neg_filter], labels[neg_filter]

        x_train1, x_test1, y_train1, y_test1 = train_test_split(pos_d, pos_l, train_size=train_ratio, random_state=random_state)
        x_train2, x_test2, y_train2, y_test2 = train_test_split(neg_d, neg_l, train_size=train_ratio, random_state=random_state)
        x_train1 = np.append(x_train1, x_train2, axis=0)
        x_test1 = np.append(x_test1, x_test2, axis=0)
        y_train1 = np.append(y_train1.to_frame(), y_train2.to_frame(), axis=0)
        y_test1 = np.append(y_test1.to_frame(), y_test2.to_frame(), axis=0)
        x_train1 = np.append(x_train1, y_train1, axis=1)
        x_test1 = np.append(x_test1, y_test1, axis=1)
        np.random.shuffle(x_train1)
        np.random.shuffle(x_test1)
        x_train1 = x_train1.astype(float)
        x_test1 = x_test1.astype(float)

        return x_train1[:, :-y_train1.shape[1]], x_test1[:, :-y_test1.shape[1]], x_train1[:, -y_train1.shape[1]:].ravel(), x_test1[:,
                                                                                                                           -y_test1.shape[
                                                                                                                               1]:].ravel()

    def train_models(self):

        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {model_name: executor.submit(model.fit, self.X_train, self.y_train) for (model_name, model) in self.models.items()}
        pass

    def predict(self, X):

        model_predictions = dict()
        for model_name, model in self.models.items():
            model_predictions[model_name] = model.predict(X)
        self.model_predictions = model_predictions
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
        return roc_auc_score(self.y_test, given_pred)

    def sens(self, given_pred):

        def youden(sens, spec):
            return sens[np.argmax(sens + spec)]

        fpr, sens, _ = roc_curve(self.y_test, given_pred)
        spec = 1 - fpr
        return youden(sens, spec)

    def spec(self, given_pred):

        def youden(sens, spec):
            return spec[np.argmax(sens + spec)]

        fpr, sens, _ = roc_curve(self.y_test, given_pred)
        spec = 1 - fpr
        return youden(sens, spec)

    def accuracy(self, given_pred):

        def distance_from_one(val):
            return abs(1 - val)

        def youden(sens, spec, th):
            return th[np.argmax(sens + spec)]

        fpr, sens, ths = roc_curve(self.y_test, given_pred)
        spec = 1 - fpr
        th = youden(sens, spec, ths)
        pred_classes = given_pred >= th
        vals = self.y_test + pred_classes
        total = 0
        for v in map(distance_from_one, vals):
            total += v
        return total / len(given_pred)


# create required directories if they do not exist
for val in CTA_data_formatter.REQ_DIRS:
    try:
        os.mkdir(val)
        print('Created directory: ' + val)
    except OSError:
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

        if (not settings['keep_cta']) and settings['keep_pet']:
            continue
        yield settings


CUSTOM_VARIABLES = {
        'passed time': CTA_data_formatter.calculate_passed_time,
        'stenosis type counts': CTA_data_formatter.count_stenosis_types,
        'events': CTA_data_formatter.combine_labels,
        'cv events': CTA_data_formatter.combine_cv_labels,
        'timed events': CTA_data_formatter.create_time_restricted_labels,
        'timed cv events': CTA_data_formatter.create_time_restricted_cv_labels,
        'min cta type': CTA_data_formatter.calculate_min_cta_type,
        'max cta type': CTA_data_formatter.calculate_max_cta_type,
        'min pet value': CTA_data_formatter.create_min_str_seg,
        'max pet value': CTA_data_formatter.create_max_str_seg
    }

if __name__ == "__main__":

    format_original_data = True
    iterate_all_options = True
    iteration_start = 0



    settings = {
        'only_pet': False,
        'timed': False,
        'keep_cta': False,
        'keep_pet': False,
        'cv': False
    }

    if format_original_data:
        cta_data = CTA_data_formatter(custom_variables=CUSTOM_VARIABLES)
        cta_data.to_csv()

    curr_iter = 1
    for opt in iter_options(settings, iteration_start):
        print(f'Starting settings no. {curr_iter} \n')
        curr_iter += 1
        cta_data = CTA_class(**opt)()
