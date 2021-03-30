import pandas as pd
import numpy as np
import string
import codecs
import datetime
import matplotlib.pyplot as plt
import feature_data


def csv_read(file_path, index_col=0):
    def auto_truncate(val):
        return val[:255]
    s = ""
    with codecs.open(file_path, 'r+', encoding='utf-8', errors='ignore') as fdata:
        s = fdata.read()
        with open('Processed_data.csv', 'w', encoding='utf-8') as f:
            f.write(s)
    return pd.read_csv('Processed_data.csv',  encoding='utf-8', converters={'label': auto_truncate}, index_col=index_col)


# Creates a copy of dataframe df and returns the copy with columns parsed based on mode
# 'quantile' checks if at least q portion of rows is not missing
# 'constant' checks if at least c rows is not missing
# ignore_list can be either None or list-like.
# if list-like, the method keeps the columns given in ignore_list regardless of how many missing values it has
def data_parse_cols(df, mode='quantile', q=0.9, c=10, ignore_list=None):
    to_return = df.copy()

    req = 0

    if mode == 'quantile':
        assert q is not None and 0 < q < 1, 'q should be strictly between 0 and 1 when mode == "quantile"'

        req = int(len(df.index)*q)
    elif mode == 'constant':
        assert c is not None and c >= 0, 'c should be greater than or equal to 0 when mode == "constant"'

        req = c

    if ignore_list is None:
        return to_return.dropna(axis='columns', thresh=req)
    else:
        ignored = to_return[ignore_list]
        to_return.drop(labels=ignore_list, axis='columns', inplace=True)

    to_return.dropna(axis='columns', thresh=req, inplace=True)

    if ignore_list is not None:
        for var in ignored.columns:
            to_return[var] = ignored[var]

    return to_return


# attempts to turn str text in dataframe to numeric values
def data_str_to_num(df, method, *args, **kwargs):

    def wrapper_function(df, *args, **kwargs):
        for row in df.index:
            for col in df.columns:
                if isinstance(df.at[row,col], str):
                    df.at[row,col] = method(df.at[row,col],*args,**kwargs)
        return df
    return wrapper_function(df, *args, **kwargs)


def data_remove_indices(correct_df, abundant_df):
    abundant_is = list()

    for i in abundant_df.index:
        if i not in correct_df.index:
            abundant_is.append(i)

    abundant_df.drop(abundant_is, inplace=True)

    pass

def cta_data_parser(str):

    zeroes = ['Male', 'Unchecked', 'confirmed not MI', 'no']
    ones = ['Female', 'Checked', 'confirmed MI', 'yes', 'yes previous smoking history', 'yes current',
            'typical AP (pressure, stress, nitrogen helps etc)']

    if str in zeroes:
        return 0
    elif str in ones:
        return 1

    start_i = -1
    end_i = len(str)
    for i, l in enumerate(str):
        if l in string.digits and start_i == -1:
            if i > 3:
                break
            start_i = i
        elif l not in string.digits and start_i != -1:
            end_i = i
            break

    if start_i == -1:
        return np.nan
    elif end_i == len(str):
        return int(str[start_i:])
    else:
        return int(str[start_i:end_i])


def data_correct_labels(df, y_labels, d_labels):

    for i,l in enumerate(y_labels):
        df.loc[df['label'].str.contains(pat=l[:9]), ['label']] = d_labels[i]

    return df


def val_in_list(df, var, val_list):
    to_return = pd.Series()
    for val in val_list:
        val_results = pd.Series(df[var] == val)
        if len(to_return) == 0:
            to_return = val_results
        else:
            to_return += val_results

    return to_return


def data_desired_functions(data, function_matrix, ignore=None):
    if ignore is None:
        ignore = []
    import sys
    to_return = data.copy()
    for f in function_matrix.index:
        func = getattr(sys.modules[__name__], 'data_' + f)
        for n in function_matrix.columns:
            if n in ignore:
                continue
            val = function_matrix.at[f, n]
            if val is not None:
                to_return = func(to_return, n, val)
    return to_return


def data_req(df, var_name, val):

    if isinstance(val, (tuple, list, set)):
        to_return = df.loc[val_in_list(df, var_name, val)]
        return to_return
    else:
        return df.loc[df[var_name] == val]


def data_missing(df, var_name, val):
    df[var_name].fillna(value=val, inplace=True)
    return df


def data_transform(df, var_name, val):

    values_list = list()
    for i, t in enumerate(val):
        if i != len(val) - 1:
            df.loc[val_in_list(df, var_name, t[0]), [var_name]] = t[1]
            values_list.append(t[1])
        else:
            df.loc[~val_in_list(df, var_name, values_list), [var_name]] = t[1]
    return df

def data_between(df, var_name, val):
    df = df.loc[val[0] <= df[var_name]]
    df = df.loc[df[var_name] <= val[1]]
    return df


def data_handle_dates_old(df, start, end):

    def date_to_num(date_str):

        if not isinstance(date_str, str):
            return np.nan
        if len(date_str) < 8:
            return np.nan

        year = month = day = None
        check_point = 0
        for i, c in enumerate(date_str):
            if c == '-':
                if i == 4:
                    try:
                        year = int(date_str[0:i])
                        check_point = i
                        if year <= 0:
                            return np.nan
                    except ValueError:
                        return np.nan
                elif i == 6 or i == 7:
                    try:
                        month = int(date_str[(check_point+1):i])
                        check_point = i
                        if month <= 0:
                            return np.nan
                    except ValueError:
                        return np.nan

        try:
            day = int(date_str[(check_point + 1):])
            if day < 0 or day > 31:
                day = 1
        except ValueError:
            day = 1

        try:
            d0 = datetime.datetime(1,1,1)
            d1 = datetime.datetime(year, month, day)
            delta = d1 - d0
            return delta.days
        except ValueError:
            return np.nan


    all_dates = list()
    if not isinstance(start, (tuple, list, set)):
        start = [start]
    if not isinstance(end, (tuple, list, set)):
        end = [end]

    all_dates.extend(start)
    all_dates.extend(end)
    for var in all_dates:
        for i in df[var].index:
            df.at[i, var] = date_to_num(df.at[i, var])
    pass

def data_handle_dates(df, start, end):

    def date_parser(dates):
        for i, date_str in enumerate(dates):
            if not isinstance(date_str, str):
                dates.iloc[i] = None
            elif len(date_str) != 10:
                dates.iloc[i] = None
            else:
                dates.iloc[i] = datetime.date(int(date_str[:4]), int(date_str[5:7]), int(date_str[8:]))
        return dates
    dates = [n for n in start]
    dates.extend([n for n in end])
    return df.loc[:, dates].apply(date_parser)


def data_passed_time(df, start, end):
    the_beginning = datetime.date(2000, 1,1)
    temp_dict = dict()
    for val in start:
        temp_dict[val] = feature_data.END_OF_FOLLOW_UP_DATE
    for val in end:
        temp_dict[val] = the_beginning

    min_dates = dict()
    max_dates = dict()
    dates_d = list()
    for id, vals in df.fillna(temp_dict).iterrows():
        min_dates[id] = vals[start].min(skipna=True)
        max_dates[id] = vals[end].max(skipna=True)
        if max_dates[id] <= min_dates[id]:
            max_dates[id] = feature_data.END_OF_FOLLOW_UP_DATE
        dates_d.append((max_dates[id] - min_dates[id]).days)

    df['Passed time'] = dates_d #todo

    pass

def data_heatmap(df):
    import matplotlib.pyplot as plt
    plt.pcolor(df)
    plt.yticks(np.arange(0.5, len(df.index)))
    plt.xticks(np.arange(0.5, len(df.columns)))
    plt.show()
    pass

def data_plot_nas(df, axis=0):

    array = df.isna().astype(int).sum(axis=axis)
    plt.hist(array)
    plt.show()

def data_plot_bar(srs):
    plt.bar(range(len(srs)), srs, tick_label=srs.index)
    plt.show()

def data_count_stenosis_types(data, n_classes=7):

    new_c_names = [f'Stenosis type {n} count' for n in range(n_classes)]

    # find the "type of stenosis" variables in the data

    first_type_name = None
    last_type_name = None
    previous_name = None

    for var_name in data.columns:
        if "type of stenosis" in var_name:
            if first_type_name is None:
                first_type_name = var_name
            else:
                previous_name = var_name
        elif first_type_name is not None:
               last_type_name = previous_name
               break
    if last_type_name is None:
        last_type_name = previous_name

    stenosis_data = data.loc[:, first_type_name:last_type_name]

    def countif(values, i):
        count = 0
        if i != 0:
            for val in values:
                if val == i:
                    count += 1
        else:
            for val in values:
                if np.isnan(val):
                    count += 1
        return count

    count_vals = dict()
    for i in range(n_classes):
        if i > 0:
            count_vals[new_c_names[i]] = stenosis_data.apply(lambda x: countif(x, i), axis=1)
            data[new_c_names[i]] = count_vals[new_c_names[i]]

    return data

def data_train_test_split(data, labels, train_ratio=3/4, random_state=None):
    from sklearn.model_selection import train_test_split
    if random_state is None:
        random_state = datetime.datetime.now().microsecond % 1000
    pos_filter, neg_filter = labels.sum(axis=1) > 0, labels.sum(axis=1) <= 0
    pos_d, pos_l, neg_d, neg_l = data[pos_filter], labels[pos_filter], data[neg_filter], labels[neg_filter]

    x_train1, x_test1, y_train1, y_test1 = train_test_split(pos_d, pos_l, train_size=train_ratio, random_state=random_state)
    x_train2, x_test2, y_train2, y_test2 = train_test_split(neg_d, neg_l, train_size=train_ratio, random_state=random_state)
    x_train1= np.append(x_train1, x_train2, axis=0)
    x_test1= np.append(x_test1, x_test2, axis=0)
    y_train1= np.append(y_train1, y_train2, axis=0)
    y_test1= np.append(y_test1, y_test2, axis=0)
    x_train1 = np.append(x_train1, y_train1, axis=1)
    x_test1 = np.append(x_test1, y_test1, axis=1)
    np.random.shuffle(x_train1)
    np.random.shuffle(x_test1)

    return x_train1[:, :-y_train1.shape[1]], x_test1[:, :-y_test1.shape[1]], x_train1[:, -y_train1.shape[1]:], x_test1[:, -y_test1.shape[1]:]

def data_pp_keras(*args):
    pass

def data_drop(data, CTAs=True, PETs=True):
    from feature_data import CTA_FEATURES
    from feature_data import PET_FEATURES
    if CTAs:
        data = data.drop(columns=CTA_FEATURES, errors='ignore')
    if PETs:
        data = data.drop(columns=PET_FEATURES, errors='ignore')


    return data
















# for module testing
if __name__ == "__main__":
    df = csv_read('CTArekisteri_DATA_2020-03-24_1051.csv')

    df2 = pd.DataFrame({'eka': [1, np.NaN, 3],
                        'toka': [3, 4, 5],
                        'kolmas': [6, 7, 8]})
    df2.to_csv('testi.csv', index=False)