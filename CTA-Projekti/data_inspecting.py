import pandas as pd
from CTA_class import PROCESSED_DATA_DIR
from scipy.stats import shapiro
import matplotlib.pyplot as plt


def mean(vals, *args):
    return vals.mean()


def standard_dev(vals, *args):
    return vals.std()


def median(vals, *args):
    return quantile(vals, 0.5)


def lower_quartile(vals, *args):
    return quantile(vals, 0.25)


def upper_quartile(vals, *args):
    return quantile(vals, 0.75)


def quantile(vals, q, *args):
    return vals.quantile(q)


def missing(vals, *args):
    return vals.isna().sum() / vals.shape[0]


data = pd.read_csv(PROCESSED_DATA_DIR + '\\Original data.csv', dtype='float64', index_col=0, header=0)
data.drop(labels='passed time', axis=1, inplace=True)
training_data = pd.read_csv("Training data.csv", dtype='float64', index_col=0, header=0)
test_data = pd.read_csv("Test data.csv", dtype='float64', index_col=0, header=0)
training_labels = pd.read_csv("Training labels.csv", dtype='float64', index_col=0, header=0)
test_labels = pd.read_csv("Test labels.csv", dtype='float64', index_col=0, header=0)

training_data = data.loc[training_data.index, list(training_data.columns) + ['early revasc', 'late revasc']]
test_data = data.loc[test_data.index, list(test_data.columns) + ['early revasc', 'late revasc']]
training_labels = data.loc[training_labels.index, training_labels.columns]
test_labels = data.loc[test_labels.index, test_labels.columns]
training = pd.concat([training_data, training_labels], axis=1)
testing = pd.concat([test_data, test_labels], axis=1)
categories = {'Training': training, 'Test':testing}
metrics = {'Mean': mean,
           'STD': standard_dev,
           'Median': median,
           'lower quartile': lower_quartile,
           'upper quartile': upper_quartile,
           'Mean / Median': lambda x: 9999,
           'STD / Interquartile Range': lambda x: 9999,
           'Missing': missing,
           'Shapiro-Wilk p-value': lambda x: 99999,

           }
cols = pd.MultiIndex.from_product([categories.keys(), metrics.keys()])
df = pd.DataFrame(index=data.columns, columns=cols, dtype='float64')


def get_interquartile_range(lower_quartiles, upper_quartiles):

    to_return = pd.Series(index = lower_quartiles.index)
    for (ind, lower), upper in zip(lower_quartiles.iteritems(), upper_quartiles):
        to_return[ind] = f'{lower} - {upper}'
    return to_return


for category, cat_data in categories.items():
    for metric, metric_func in metrics.items():
        df.loc[:, (category, metric)] = metric_func(cat_data)
    for col, vals in cat_data.iteritems():
        _, p_val = shapiro(vals)
        df.loc[col, (category, 'Shapiro-Wilk p-value')] = p_val
        df.loc[col, (category, 'Mean / Median')] = df.loc[col, (category, 'Median')] if p_val <= 0.05 else df.loc[col, (category, 'Mean')]
        lower_q = df.at[col, (category, "lower quartile")]
        upper_q = df.at[col, (category, "upper quartile")]
        df.loc[col, (category, 'STD / Interquartile Range')] = f'{lower_q} - {upper_q}' if p_val <= 0.05 else df.loc[col, (category, 'STD')]


df = df.round(decimals=3)
df = df.loc[:, (slice(None), ['Mean / Median', 'STD / Interquartile Range', 'Missing'])]
df_train = df['Training']
df_test = df['Test']
df.to_csv('Data summary.csv')
df_train.to_csv('Training data summary.csv')
df_test.to_csv('Test data summary.csv')
