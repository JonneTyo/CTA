import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import shapiro
import CTA_class


def get_event_from_file_name(file_name):
    label = 'event'
    label = 'cv ' + label if 'cv' in file_name else label
    label = 'timed ' + label if 'timed' in file_name else label
    return label

#returns a dic of all results with filename as keys and dataframe as items
def get_result_data(path='\\Results'):

    to_return = dict()
    for f in os.listdir(os.getcwd() + path):
        if f.endswith(".csv"):
            df = pd.read_csv(os.getcwd() + path + '\\' + f, index_col=0)
            to_return[f[:-4]] = df
    return to_return

#returns a dataframe with multi-index indices that contains data for each model-label pair
def get_best_models(dfs, measure_by='sensitivity', split_by=None):
    labels = ['event', 'cv event', 'timed event', 'timed cv event']
    to_return = pd.DataFrame(index=labels, columns=CTA_class.CTA_class.METRICS.values())
    to_return['Model'] = ""
    to_return['Model Version'] = ""
    to_return[measure_by] = -np.inf
    for key, item in dfs.items():
        label = get_event_from_file_name(key)
        for ind, row in item.iterrows():
            if to_return.at[label, measure_by] < row[measure_by]:
                to_return.loc[label, row.index] = row
                to_return['Model'] = ind
                to_return['Model Version'] = key
    return to_return

# creates .csv files where each row has the results of the best data selection method
def get_best_results():
    all_results = get_result_data()
    split_by = 'only_PET'

    for m in ['Sensitivity', 'Test AUC']:
        best_results = get_best_models(all_results, measure_by=m, split_by=split_by)
        best_results.to_csv(os.getcwd() + f'\\Result analysis\\best_results_{m}.csv')
    print(best_results)
    pass


def plot_hist_passed_time_events(data, labels):
    x = data.loc[:, 'passed time'].dropna()/365
    x = x.astype(int)
    to_return = dict()

    for l in labels:
        plt.clf()
        pos_lbls = data[l] >= 1
        x_pos = x.loc[pos_lbls]
        vals, *_ = plt.hist(x_pos, rwidth=0.8, bins=[n for n in range(max(x_pos) + 1)])
        plt.xlabel('Years passed since start of follow-up')
        plt.title(f'{l}, N={len(x_pos)}')
        plt.xticks(np.arange(0.1, (max(x_pos) + 1), step=1), labels=[n for n in range(max(x_pos) + 1)])
        plt.savefig(f'Plots\\hist_passed_time_{l}')
        to_return[l] = vals
    return to_return

# plots a histogram where x-axis shows the number of missing variables and y the number of patients.
def plot_hist_missing_vars(df):
    na_n = df.isna().sum(axis=1)
    plt.clf()
    plt.xlabel("# of missing variables")
    plt.ylabel("# patients")
    plt.title(f'# of patients = {len(df.index)}')
    plt.hist(na_n, bins=[10 * i for i in range(int(na_n.min() / 10), int(na_n.max() / 10 + 1))])

    plt.savefig('Plots\\hist_of_n_patients_with_x_missing_vars_new.png')
    pass

# Calculates the differences between results when a setting is turned on/off
def get_setting_differences():
    settings = ['keep_cta', 'keep_pet']

    to_return = pd.DataFrame(index=settings, columns=CTA_class.CTA_class.METRICS.values())
    for setting in settings:
        df_setting = pd.Series(data=0, index=CTA_class.CTA_class.METRICS.values(), dtype='float64')
        df_no_setting = pd.Series(data=0, index=CTA_class.CTA_class.METRICS.values(), dtype='float64')
        n = 0
        for f in os.listdir(os.getcwd() + '\\Results'):
            if 'timed' in f or 'cv' not in f:
                continue
            n += 1
            temp = pd.read_csv(os.getcwd() + '\\Results\\' + f, index_col=0, header=0).mean(axis=0)
            if setting in f:
                df_setting += temp
            else:
                df_no_setting += temp
        if setting == 'keep_cta':
            df_no_setting = 2*df_no_setting
        if setting == 'keep_pet':
            df_setting = 2*df_setting
        temp = (df_setting - df_no_setting)/n
        to_return.loc[setting, :] = temp
    to_return.to_csv(os.getcwd() + '\\Result analysis\\settings_differences_no_timed_cv.csv')





if __name__ == "__main__":
    settings = {
        'only_pet': True,
        'timed': False,
        'keep_cta': True,
        'keep_pet': True,
        'cv': False
    }


    get_best_results()


    pass
