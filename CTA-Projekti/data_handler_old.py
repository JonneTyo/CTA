import matplotlib
import matplotlib.pyplot as plt
import random
from cta_maths import *
from sklearn.model_selection import KFold


# Attempts to plot a heatmap of given data
def data_heatmap(data, rows=None, x_labels=None):
    yticks = np.arange(len(data))
    x_names = data[0]
    if rows is not None:
        yticks = rows
    if x_labels is not None:
        x_names = x_labels
    fig, axes = plt.subplots()
    im = axes.imshow(data[1:])

    axes.set_xticks(np.arange(len(data[0])))
    axes.set_yticks(yticks)
    axes.set_xticklabels(x_names)

    plt.setp(axes.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    fig.tight_layout()
    plt.show()
    pass


# Function returns a binary np.array with the shape of data where 0 is given to the datapoints with value None and
# 1 otherwise
def data_missing_values(data):
    to_return = np.zeros((len(data), len(data[0])))

    for row in range(len(data)):
        for value in range(len(data[0])):

            if data[row][value] is not None:
                to_return[row][value] = 1

    return to_return


# Function separates data into 2 lists, training and testing, with ratio amount of the original data being in the
# training set and returns the lists.
def data_training_set(x, y_labels, ratio=0.67, seed=None):
    data = x[1:]
    labels = y_labels[1:]
    if ratio < 0 or ratio > 1:
        raise Exception('Argument ratio should be between 0 and 1')
    if seed is not None:
        random.seed(a=seed)
    data_amount = len(data)
    training_amount = int(data_amount * ratio)

    training_set_ids = random.sample(range(data_amount), training_amount)
    training_set = []
    testing_set = []
    training_labels = []
    testing_labels = []

    for i in range(data_amount):
        if i in training_set_ids:
            training_set.append(data[i])
            training_labels.append(labels[i])
        else:
            testing_set.append(data[i])
            testing_labels.append(labels[i])

    return training_set, testing_set, training_labels, testing_labels


# Returns a list of rows listed in iterable from a given reader. Argument iterable must be sorted in ascending order!!!
def data_lines(data, iterable):
    to_return = []
    row_count = len(data)

    for i in iterable:
        if row_count > i >= 0:
            to_return.append(data[i])
        else:
            raise Exception('Unsuitable row number in iterable: {}'.format(i))

    if len(to_return) == 0:
        print('Warning: data_lines returned no lines')
    return to_return


# Returns a list of the columns listed in iterable from data
def data_columns(data, iterable):
    to_return = []
    columns_n = len(data[0])
    for i in iterable:
        if i in data[0]:
            col_i = data[0].index(i)
            column = []
            for line in data:
                column.append(line[col_i])
            to_return.append(column)
    if len(to_return) == 1:
        return to_return[0]
    else:
        return to_return


# Returns a list of lines in the data where the columns in iterable are not None
def data_require_col(data, iterable):
    to_return = []
    for i in iterable:
        if i < 0 or i >= len(data[0]):
            raise Exception('Invalid column in argument iterable: {}'.format(i))
        for row in data:
            if row[i] is not None:
                to_return.append(row)

    return to_return


# Removes columns from data based on mode chosen, keeping the columns in ignore
# q must not be None when mode is quantile
# c must not be None when mode is constant
# quantile mode removes the columns where the proportion of Nones is more than q
# constant mode removes the columns where there are less than c rows of data
def data_parse(x, q=None, c=None, mode='quantile', ignore=None):
    data = x
    modes = 'quantile, constant'
    to_return = []

    columns = data_columns(data, range(len(data[0])))
    pop_list = []
    iterable = []
    if mode == 'quantile':
        if q is None:
            raise Exception('Argument q must not be None when mode is quantile')
        else:

            # Determine which columns should be kept
            for col in columns:

                # This column is kept
                if maths_sum_Nones(col) <= q * len(col):
                    pop_list.append(False)

                # This column is popped
                else:
                    pop_list.append(True)

    elif mode == 'constant':
        if c is None:
            raise Exception('Argument c must not be None when mode is constant')
        else:
            for col in columns:
                if len(col) - maths_sum_Nones(col) >= c:
                    pop_list.append(False)
                else:
                    pop_list.append(True)

    else:
        raise Exception('Incorrect mode {}. Argument mode must be one of the following: '.format(mode), modes)

    for i in range(len(pop_list)):
        if pop_list[i]:
            iterable.append(i)

    if ignore is not None:
        for i in ignore:
            if i in iterable:
                iterable.remove(i)

    to_return = data_pop_columns(data, iterable=iterable)

    return to_return


# Function pops the columns given in iterable from the data
# ONLY USE WITH COMPLETE DATA!!!
def data_pop_columns(x, iterable):
    to_return = []
    i_list = []

    for i in iterable:
        if i in x[0]:
            i_list.append(x[0].index(i))

    i_list.sort()
    for row in x:
        to_reduce = 0
        for i in i_list:

            row.pop(i - to_reduce)
            to_reduce += 1
        to_return.append(row)
    return to_return


# Returns the amount of features each row in the data has
def data_feature_amount(data):
    to_return = []

    for row in data:
        to_return.append(len(row) - maths_sum_Nones(row))

    return to_return


# Plots a histogram of the data
def data_histogram(data, intervals='auto'):
    n, bins, patches = plt.hist(x=data, bins=intervals)

    plt.show()

    pass


# Returns the rows in data with only the columns that all the rows have as not None
def data_intersection(x):
    data = x
    itera = set()

    for row in data:
        for i in range(len(row)):
            if row[i] is None:
                itera.add(i)

    return data_pop_columns(data, itera)


# A function for testing other functions
def data_testing_features(data):
    f_min = len(data[0])
    f_max = len(data[0])

    for row in data:
        if len(row) < f_min:
            f_min = len(row)
        elif len(row) > f_max:
            f_max = len(row)
    print('Lowest amount of features: ', str(f_min))
    print('Highest amount of features: ', str(f_max))

    return f_min == f_max


# Returns the corresponding indices from label_list based on the feature_label_names
# feature_alt_names can be passed as an alternative list of names
def data_find_feature_cols(label_list, feature_label_names, feature_alt_names=None):

    index_set1 = {label_list.index(name) for name in feature_label_names}
    index_set2 = {}
    if feature_alt_names is not None:
        index_set2 = {label_list.index(name) for name in feature_alt_names}
    index_set1.union(index_set2)
    index_set1 = list(index_set1)
    index_set1.sort()
    return index_set1

def data_as_matrix(data):


    if type(data[0]) == list:
        x = np.zeros((len(data), len(data[0])))
        is_matrix = True
    else:
        x = np.zeros((len(data), 1))
        is_matrix = False

    for row in range(len(data)):
        if is_matrix:
            for col in range(len(data[row])):
                x[row, col] = data[row][col]
        else:
            x[row, :] = data[row]

    return x


# Returns a list of length n which has tuples containing the training indices and testing indices of that fold
# The length of training indices will be ratio*data amount
def data_kfold(X, n=2, ratio=0.67, random_state=None):
    if ratio <= 0 or ratio >= 1:
        raise Exception('Argument ratio must be between 0 and 1. Argument given: {}'.format(ratio))
    else:
        n_train = int(X.shape[0]*ratio)
        n_test = X.shape[0] - n_train
        n_total = range(X.shape[0])

    random.seed(a=random_state)
    to_return = []
    for fold in range(n):
        train_i = random.sample(n_total, n_train)
        test_i = []
        for i in n_total:
            if i not in train_i:
                test_i.append(i)
        to_return.append((train_i, test_i))
    return to_return

def data_convert_nans(X, target_conversion=-1):
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            if np.isnan(X[i,j]):
                X[i,j]=target_conversion
    return X


