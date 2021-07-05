import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import shapiro
import CTA_class

# generates the used label based on the filename
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
    x = data.loc[:, 'passed time'].fillna(-1)/365
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
        n_setting_files, n_no_setting_files = 0, 0
        for f in os.listdir(os.getcwd() + '\\Results'):

            if setting == 'keep_pet':
                if 'keep_cta' not in f:
                    continue

            if 'timed' not in f or 'cv' not in f:
                continue
            temp = pd.read_csv(os.getcwd() + '\\Results\\' + f, index_col=0, header=0).mean(axis=0)
            if setting in f:
                df_setting += temp
                n_setting_files += 1
            else:
                df_no_setting += temp
                n_no_setting_files += 1

        df_no_setting = df_no_setting*n_setting_files/n_no_setting_files
        temp = (df_setting - df_no_setting)/(n_setting_files + n_no_setting_files)
        to_return.loc[setting, :] = temp
    to_return.to_csv(os.getcwd() + '\\Result analysis\\settings_differences_timed_cv.csv')


def plot_hist_passed_time(data):
    data = data.dropna(axis=0, how='any', subset=['passed time'])
    x = data['passed time']/365
    x = x.astype(int)
    plt.clf()
    plt.hist(x, rwidth=0.8, bins=[n for n in range(max(x) + 1)])
    plt.xlabel('Years passed since start of follow-up')
    plt.xticks(np.arange(0.1, (max(x) + 1), step=1), labels=[n for n in range(max(x) + 1)])
    plt.savefig(f'Plots\\hist_passed_time.png')
    pass


def iterate_results(results_dir=CTA_class.RESULTS_DIR):
    for f in os.listdir(results_dir):
        if f.endswith(".csv"):
            yield f, pd.read_csv(results_dir + '\\' + f, index_col=0, header=0)

def plot_results_by_time(min_time=1, max_time=8, include_unlimited_time=True, metric=None, measure_by='Test AUC'):

    results = {'basic': pd.DataFrame(index=[n for n in range(min_time, max_time + int(include_unlimited_time) + 1)]),
               'cta': pd.DataFrame(index=[n for n in range(min_time, max_time + int(include_unlimited_time) + 1)]),
               'pet': pd.DataFrame(index=[n for n in range(min_time, max_time + int(include_unlimited_time) + 1)]),
               'cta and pet': pd.DataFrame(index=[n for n in range(min_time, max_time + int(include_unlimited_time) + 1)])}
    for dir in [CTA_class.RESULTS_PET_DIR, CTA_class.RESULTS_DIR]:
        for file_name, data in iterate_results(dir):

            results_category = None
            results_index = max_time + 1 if include_unlimited_time else 0

            if ('keep_pet' in file_name) and ('cta' not in file_name):
                results_category = 'pet'
            elif ('cta' in file_name) and ('keep_pet' not in file_name):
                results_category = 'cta'
            elif ('cta' in file_name) and ('keep_pet' in file_name):
                results_category = 'cta and pet'
            else:
                results_category = 'basic'

            if 'years' not in file_name:
                if not include_unlimited_time:
                    continue
            else:
                results_index = int(file_name[-11])

            best_model = data.idxmax()[measure_by]
            if metric:
                results[results_category].at[results_index, metric] = data.at[best_model, metric]
            else:
                for metr in data.columns:
                    results[results_category].at[results_index, metr] = data.at[best_model, metr]

        results['PET and CTA difference'] = results['pet'] - results['cta']
        results['Additional value of PET'] = results['cta and pet'] - results['cta']
        for key, df in results.items():
            fig, ax = plt.subplots()
            plt.grid(True)
            for metr in df.columns:
                ax.scatter([n for n in range(min_time, len(df.index) +1)], df[metr])
                ax.plot(df[metr], label=metr)
            ax.legend()
            plt.title(key)
            ticks = [n for n in range(min_time, max_time + 1)]
            if include_unlimited_time:
                ticks.append("No limit")
            plt.xticks([n for n in range(min_time, max_time + int(include_unlimited_time) + 1)], labels=ticks)
            plt.xlabel("Maximum observed time")
            if key not in ('Additional value of PET', 'PET and CTA difference'):
                plt.ylim((0.5, 1))
            else:
                plt.ylim((-0.1, 0.1))
            only_pet_txt = '_only_pet' if 'only pet' in dir else ""
            plt.savefig(os.getcwd() + f"\\Plots\\results_{key}{only_pet_txt}.png")


if __name__ == "__main__":

    plot_results_by_time()

    pass
