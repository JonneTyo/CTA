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

def set_fig_size(fig, left: float=0.1, right: float=0.9, top: float=0.95, bottom: float=0.05, size: tuple=(12.4, 9.6)):
    fig.subplotpars.left = left
    fig.subplotpars.right = right
    fig.subplotpars.top = top
    fig.subplotpars.bottom = bottom
    fig.set_size_inches(size[0], size[1])
    return fig

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


def plot_hist_PCA_values():
    from CTA_class import CTA_class
    from scipy.stats import shapiro
    from CTA_maths import linear_map
    settings = {'cta': False,
                'pet': False,
                'basic': False,
                'only_pet': False,
                'pca': True}

    for t in [None, 2, 3]:
        cta_data = CTA_class(time=t, **settings)
        data = cta_data.data_fill_na
        labels = cta_data.labels

        columns = ['PCA1', 'PCA2']
        data.columns = columns
        data['PCA2 abs'] = abs(data['PCA2'] - data['PCA2'].median())
        columns.append('PCA2 abs')
        for col in columns:
            plt.clf()
            fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
            print(f'Min value for {col}: {data[col].min()}')
            bins = [n for n in range(int(data[col].min() - 1), int(data[col].max() + 1))]
            #scaler = linear_map((-1, 1), (product[col].min(), product[col].max()))
            #vals = product[col].apply(scaler)
            median = data[col].median()
            vals_below_zero = data[col].loc[data[col] <= median]
            vals_above_zero = data[col].loc[data[col] > median]
            labels_below_zero = labels.loc[vals_below_zero.index].sum()
            labels_above_zero = labels.loc[vals_above_zero.index].sum()

            ax1.hist([vals_below_zero, vals_above_zero], bins=bins, rwidth=0.95, histtype='barstacked')
            ax1.legend([f'{labels_below_zero/len(vals_below_zero)*100:.1f}', f'{labels_above_zero/len(vals_above_zero)*100:.1f}'],
                       title=f'Event occurrence % within {t if t else "unrestricted"} years', loc='upper right')
            ax1.set_xlabel(f'{col} values')
            _, p_val = shapiro(data[col])
            ax1.set_xticks(bins)
            ax1.grid(True)

            lower_quartile_val = np.quantile(data[col], q=0.25)
            upper_quartile_val = np.quantile(data[col], q=0.75)
            lower_quartile = data[col].loc[data[col] <= lower_quartile_val]
            middle = data[col].loc[(lower_quartile_val < data[col]) & (data[col] < upper_quartile_val)]
            upper_quartile = data[col].loc[data[col] >= upper_quartile_val]
            labels_lower = labels.loc[lower_quartile.index]
            labels_middle = labels.loc[middle.index]
            labels_upper = labels.loc[upper_quartile.index]

            n, *_ = ax2.hist([lower_quartile, middle, upper_quartile], bins=bins, rwidth=0.95, histtype='barstacked')
            ax2.legend([f'{labels_lower.sum()/ len(lower_quartile) * 100:.1f}', f'{labels_middle.sum()/ len(middle) * 100:.1f}',
                        f'{labels_upper.sum()/ len(upper_quartile)*100:.1f}'],
                       title=f'Event occurrence % within {t if t else "unrestricted"} years', loc='upper right')
            ax2.set_xticks(bins)
            ax2.set_xlabel(f'{col} values')
            ax2.grid(True)
            set_fig_size(fig)
            fig.suptitle(f'Median: {data[col].median():.2f},  interquartile range: {lower_quartile_val:.2f} - {upper_quartile_val:.2f}')
            #plt.show()
            fig.savefig(os.getcwd() + f'\\Plots\\{col}_histogram_{t if t else "unrestricted"}_years.png', dpi=600)

    pass


def plot_scatter_PCA_values():
    from CTA_class import CTA_class
    from sklearn.metrics import roc_curve

    settings = {
        'pca': True,
        'only_pet': False
    }
    for t in [None, 2, 3]:
        CTA_class.MODELS.append('LogReg')
        cta_data = CTA_class(time=t, **settings)

        cta_data(write_to_csv=False)
        cta_data.predict(cta_data.data_fill_na)
        predictions = cta_data.model_predictions['LogReg']
        fpr, sens, ths = roc_curve(cta_data.labels.astype(float).to_numpy(), predictions, pos_label=1)
        spec = 1 - fpr
        J_val = ths[np.argmax(spec + sens)]
        predictions = (predictions >= J_val).astype(int)
        data = cta_data.data
        predictions = pd.Series(predictions, index=data.index)
        fig, axes = plt.subplots()
        pred_pos = predictions[cta_data.labels >= 1]
        pred_neg = predictions[cta_data.labels < 1]
        cols_pos = ['red' if n == 1 else 'white' for n in pred_pos]
        cols_neg = ['blue' if n == 1 else 'white' for n in pred_neg]
        positives = data.loc[cta_data.labels >= 1]
        negatives = data.loc[cta_data.labels < 1]
        bin_size = 4
        bins = np.arange(min(data['PCA1'].min(), data['PCA2'].min()), max(data['PCA1'].max(), data['PCA2'].max()) + bin_size, bin_size)
        hist, *_ = axes.hist2d(x=data['PCA1'], y=data['PCA2'], bins=bins)
        hist_pos, *_ = axes.hist2d(x=positives['PCA1'], y=positives['PCA2'], bins=bins)
        fig, axes = plt.subplots()
        hist = np.rot90(hist, 1)
        hist_pos = np.rot90(hist_pos, 1)
        ratio_heatmap = np.nan_to_num(hist_pos/hist)
        axes.scatter(x=negatives['PCA1'], y=negatives['PCA2'], label='No event', s=10, c=cols_neg, marker='o', edgecolors='blue')
        axes.scatter(x=positives['PCA1'], y=positives['PCA2'], label='Event', s=10, c=cols_pos, edgecolors='red')
        axes.set_xlabel('PCA1')
        axes.set_ylabel('PCA2')
        #axes.plot([2, 2], [-1, 7], c='black')
        #axes.plot([-1, 5.5], [3, 3], c='black')
        axes.set_ylim((data['PCA2'].min(), data['PCA2'].max()))
        axes.set_xlim((data['PCA1'].min(), data['PCA1'].max()))
        #axes.set_aspect('equal')
        axes.legend(loc='lower right', fontsize='x-large')
        axes.set_aspect('equal')
        im = axes.imshow(ratio_heatmap, cmap=plt.cm.Reds, vmin=0, vmax=1)
        im.set_extent((bins[0], bins[-1], bins[0], bins[-1]))
        axes.set_title(f'Heatmap of event occurrences{f" within {t} years" if t else ""} relative to number of patients', fontsize='x-large')
        plt.colorbar(im, ax=axes, shrink=0.7)
        set_fig_size(fig)
        plt.show()
        #plt.savefig(os.getcwd() + f'\\Plots\\PCA_event_occurrence_heatmap_no_abs_{t}_years.png', dpi=600)
    pass

def plot_PCA_vs_no_pet():
    from matplotlib import cm
    from matplotlib import patches
    metrics = ['Test AUC', 'Sensitivity']
    times = [2, 3, 4, 8, None]
    categories = ['with PET', 'without PET']
    index = pd.MultiIndex.from_product([categories, metrics])
    results = pd.DataFrame(index=index, columns=times)

    for t in times:
        pca_result_file = os.getcwd() + f'\\Results\\results_pca_basic_{t}_years.csv' if t is not None else os.getcwd() + f'\\Results\\results_pca_basic.csv'
        ctapca_result_file = os.getcwd() + f'\\Results\\results_basic_ctapca_{t}_years.csv' if t is not None else os.getcwd() + f'\\Results\\results_basic_ctapca.csv'
        dfs = dict()
        dfs['with PET'] = pd.read_csv(pca_result_file, index_col=0)
        dfs['without PET'] = pd.read_csv(ctapca_result_file, index_col=0)
        for metric in metrics:
            for cat in categories:
                results.at[(cat, metric), t] = dfs[cat].at['lasso', metric]

    fig, axes = plt.subplots()
    axes.set_xticks([n for n in np.arange(times[0], times[-2] + 2)])
    axes.set_xticklabels([str(n)*(n in times) for n in np.arange(times[0], times[-2] + 2)])
    axes.set_xlim((times[0] - 0.5, times[-2] + 1.5))
    axes.set_ylim((0.5, 1))
    axes.set_xlabel("Observation years")

    cols = {'with PET': 'Blues',
            'without PET': 'Oranges'}
    alphas = {'Test AUC': 0.5,
              'Sensitivity': 0.2}
    linestyles = {'Test AUC': '-',
                  'Sensitivity': '--'}

    color_legend = list()
    line_legend = list()

    for (cat, metric), values in results.iterrows():
        cmap = plt.get_cmap(cols[cat])
        patch = patches.Patch(color=cmap(alphas[metric]), label=metric)
        color_legend.append(patch)
        line, = axes.plot([i if i is not None else times[-2] + 1 for i in times], values, c=cmap(alphas[metric]), label=cat + ' ' + metric, ls=linestyles[metric])
        axes.scatter([i if i is not None else times[-2] + 1 for i in times], values, c=cmap(alphas[metric]), s=20)

        line_legend.append(line)
    first_legend = axes.legend(handles=line_legend, loc='upper right')
    plt.gca().add_artist(first_legend)
    plt.title("Results of Lasso model")
    set_fig_size(fig)
    plt.savefig(os.getcwd() + f'\\Plots\\PCA_vs_CTA_results.png', dpi=600)

    pass


if __name__ == "__main__":

    #plot_hist_PCA_values()
    #plot_scatter_PCA_values()
    plot_PCA_vs_no_pet()

    pass
