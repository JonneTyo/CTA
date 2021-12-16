import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import math
from scipy.stats import shapiro
import CTA_class

# Helper function for setting the size of figures in matplotlib
def set_fig_size(fig, left: float=0.1, right: float=0.9, top: float=0.95, bottom: float=0.05, size: tuple=(12.4, 9.6)):
    fig.subplotpars.left = left
    fig.subplotpars.right = right
    fig.subplotpars.top = top
    fig.subplotpars.bottom = bottom
    fig.set_size_inches(size[0], size[1])
    return fig


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


# loop through the result files and for each category in the dict results store the best results for each year based on Test AUC score.
def get_best_test_auc_scores(df, only_pet_patients, use_pet_data, analyzer, times, min_time, max_time):

    for t in times:

        requires = []
        requires_not = []
        contains_pet_data_string = "only_pet_pet" if only_pet_patients else "pet"

        if use_pet_data:
            requires.extend([contains_pet_data_string, "_pca"])
        else:
            requires_not.extend([contains_pet_data_string, "_pca"])

        for data, file_name in analyzer.gen_results(require=requires, require_not=requires_not, only_pet=only_pet_patients):

            if min_time <= t <= max_time:
                if not str(t) in file_name:
                    continue
            elif "years" in file_name:
                continue
            values = data.loc[:, "all"]
            values = values.sort_values(by=['Test AUC', 'Sensitivity'], ascending=False)
            if not (df.shape[1] > 0):
                for ind, val in values.iloc[0, :].iteritems():
                    df[ind] = 0
            if values.iloc[0, 0] >= df.loc[t, "Test AUC"]:
                df.loc[t, :] = values.iloc[0, :]
    return df

# Calculates the best model based on the mean Test AUC-score and returns a dataframe,
# with prediction year as indices and the training/fixed year on columns and the test_AUC score as the values.
def get_best_overall_model(df, only_pet_patients, use_pet_data, analyzer, times, min_time, max_time, use_fixed_years=False):

    # Setting up variables for the analyzer.gen_results method
    requires = []
    requires_not = []
    contains_pet_data_string = "only_pet_pet" if only_pet_patients else "pet"
    print("USING ONLY PET PATIENTS: " + str(only_pet_patients))
    print("USING PET DATA: " + str(use_pet_data))
    if use_pet_data:
        requires.extend([contains_pet_data_string, "_pca"])
    else:
        requires_not.extend([contains_pet_data_string, "_pca"])

    test_auc_values = pd.DataFrame(columns=times)
    for data, file_name in analyzer.gen_results(require=requires, require_not=requires_not, only_pet=only_pet_patients):

        # Figuring out the observation year and file name based on the file name
        # If the fixed year and observation year don't match, move to the next file
        t = None
        file_ind = file_name[:file_name.index("_event")]
        if "_event_" in file_name:
            if "None" not in file_name:
                continue
            else:
                t = times[-1]
        elif "event " in file_name:
            if "None" in file_name:
                continue
            if int(file_name[file_name.index(" years") - 1]) == int(file_name[-5]):
                t = int(file_name[-5])
            else:
                continue

        # save test AUC values of data in to the correct column (by time) if an index row has been created for this row, else append
        if ("lasso", file_ind) in test_auc_values.index:
            test_auc_values.loc[(slice(None), file_ind), t] = data[("all", "Test AUC")].values
        else:
            multi_index = pd.MultiIndex.from_product([data.index, [file_ind]], names=["Models", "Variables"])
            temp = pd.DataFrame(index=multi_index, columns=test_auc_values.columns, dtype='float64')
            temp[t] = data[("all", "Test AUC")].values
            for _, values in temp.iterrows():
                test_auc_values = test_auc_values.append(values)
                test_auc_values.index = pd.MultiIndex.from_tuples(test_auc_values.index)

    # calculate the Mean AUC values across all observation times and sort the dataframe based on these values
    test_auc_values["Mean AUC"] = test_auc_values.mean(axis=1)
    test_auc_values.sort_values("Mean AUC", inplace=True, ascending=False)
    test_auc_values.reset_index(inplace=True)

    # Pick the model with the highest Mean AUC and return a dataframe containing all the results of that model for every observation time

    file_ind = test_auc_values.iloc[0, 1]
    print("Best variables: " + file_ind)
    model_ind = test_auc_values.iloc[0, 0]
    print("Best model: " + model_ind)

    for data, file_name in analyzer.gen_results(require=(file_ind), only_pet=only_pet_patients):

        if not use_fixed_years:
            # Figuring out the observation year and file name based on the file name
            # If the fixed year and observation year don't match, move to the next file
            t = None
            if "None" in file_name:
                if "_event_" not in file_name:
                    continue
                else:
                    t = times[-1]
            elif "event " in file_name:
                if int(file_name[file_name.index(" years") - 1]) == int(file_name[-5]):
                    t = int(file_name[-5])
                else:
                    continue

                for ind, val in data["all"].loc[model_ind, :].items():
                    if ind not in df.columns:
                        df[ind] = 0
                        df.loc[t, ind] = val
        else:

            # Figuring out the observation year and fixed year based on the file name
            t = None
            t_fixed = None

            if "None" in file_name:
                t_fixed = times[-1]
            else:
                t_fixed = int(file_name[-5])

            if "_event_" in file_name:
                t = times[-1]
            else:
                t = int(file_name[file_name.index(" years") - 1])

            val = data["all"].at[model_ind, "Test AUC"]
            if t_fixed not in df.columns:
                df[t_fixed] = 0
            df.loc[t, t_fixed] = val
    return df


# Generator that iterates through all the files in results_dir and yields the file name and pd.dataframe if the file is a .csv
def iterate_results(results_dir=CTA_class.RESULTS_DIR):
    for f in os.listdir(results_dir):
        if f.endswith(".csv"):
            yield f, pd.read_csv(results_dir + '\\' + f, index_col=0, header=[0,1])

# takes in a dataframe with different models on indices and returns the col
# with best linearly weighted averages
def best_model_linear_weighting(df: pd.DataFrame):

    to_return = pd.DataFrame(columns=[1])
    weights = np.linspace(2, 1, num=8)
    weights = np.append(weights, 1)
    max_val = 0
    for ind, col in df.iteritems():
        curr_val = (weights*col).sum()
        if max_val < curr_val:
            max_val = curr_val
            to_return.columns = [ind]
            to_return[ind] = col

    return to_return


def plot_results_by_time(min_time=1, max_time=8, include_unlimited_time=True, only_one_model=True):

    results = {'all patients using pet': pd.DataFrame(index=[n for n in range(min_time, max_time + int(include_unlimited_time) + 1)]),
               'all patients no pet': pd.DataFrame(index=[n for n in range(min_time, max_time + int(include_unlimited_time) + 1)]),
               'pet patients using pet': pd.DataFrame(index=[n for n in range(min_time, max_time + int(include_unlimited_time) + 1)]),
               'pet patients no pet': pd.DataFrame(index=[n for n in range(min_time, max_time + int(include_unlimited_time) + 1)])}

    from CTA_analyzer import CTA_analyzer

    RESULTS_DIR = os.getcwd() + '\\Final results'
    ONLY_PET_RESULTS_DIR = os.getcwd() + '\\Final results only pet'
    OUTPUT_DIR = os.getcwd() + '\\Result analysis'

    analyzer = CTA_analyzer(RESULTS_DIR, ONLY_PET_RESULTS_DIR, OUTPUT_DIR)
    #revasc_options = ['all', 'no revasc', 'with revasc']
    times = [n for n in range(min_time, max_time+1)]
    if include_unlimited_time:
        times.append(max_time + 2)
        for df in results.values():
            df.rename(index={max_time + 1: max_time + 2}, inplace=True)

    for only_pet_patients in [True, False]:

        for use_pet_data in [True, False]:
            result_category = "pet patients" if only_pet_patients else "all patients"
            result_category = result_category + " using pet" if use_pet_data else result_category + " no pet"

            # Choose one method to get the results.
            results[result_category] = get_best_overall_model(results[result_category], only_pet_patients, use_pet_data, analyzer, times, min_time, max_time, use_fixed_years=True)
            #results[result_category] = get_best_test_auc_scores(results[result_category], only_pet_patients, use_pet_data, analyzer, times, min_time, max_time)



        fig, axes = plt.subplots(nrows=1, ncols=3)
        fig.set_size_inches(19.2, 6.4)
        ax_pet, ax_no_pet, ax_diff = axes

        suffix = "pet " if only_pet_patients else "all "
        df_pet = suffix + "patients using pet"
        df_pet = results[df_pet]
        df_no_pet = suffix + "patients no pet"
        df_no_pet = results[df_no_pet]


        if only_one_model:
            df_pet = best_model_linear_weighting(df_pet)
            df_no_pet = best_model_linear_weighting(df_no_pet)

        for metr_pet, metr_no_pet in zip(df_pet.columns, df_no_pet.columns):
            ax_pet.scatter(df_pet.index, df_pet[metr_pet])
            ax_no_pet.scatter(df_no_pet.index, df_no_pet[metr_no_pet])
            ax_pet.plot(df_pet.index, df_pet[metr_pet], label=metr_pet)
            ax_no_pet.plot(df_no_pet.index, df_no_pet[metr_no_pet], label=metr_no_pet)
            ax_diff.scatter(df_pet.index, df_pet[metr_pet] - df_no_pet[metr_no_pet])
            ax_diff.plot(df_pet.index, df_pet[metr_pet] - df_no_pet[metr_no_pet], label=f'{metr_pet} - {metr_no_pet}')

        ticks = [n if n != max_time + 1 else "" for n in range(min_time, max_time + 1 + int(include_unlimited_time))]
        if include_unlimited_time:
            ticks.append("Unrestricted")

        ax_titles = ["Using PET data", "Not using PET data", "Added value of PET data"]
        for ax, title in zip(axes, ax_titles):
            ax.legend()
            ax.set_xlabel("Maximum observed time")
            ax.set_xticks([n for n in range(min_time, max_time + int(include_unlimited_time) + 2)])
            ax.set_xticklabels(ticks)
            ax.grid(True)
            if ax != ax_diff:
                ax.set_ylim((0.4, 1))
            else:
                ax.set_ylim((-0.2, 0.2))
            ax.set_title(title)


        fig.suptitle("Model results when using all patients" if not only_pet_patients else "Model results when using only PET patients")
        plt.savefig(os.getcwd() + f"\\Plots\\best_TRAINING_model_{'only pet patients' if only_pet_patients else 'all patients'}_fixed_years.png")


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


def plot_results_grouped_by_revasc(min_time=1, max_time=8, include_unlimited_time=True, measure_by='Test AUC'):

    indices = pd.Index([n for n in range(min_time, max_time + int(include_unlimited_time) + 1)])
    revasc_options = ['all', 'no revasc', 'with revasc']
    results_dirs = [CTA_class.RESULTS_DIR, CTA_class.RESULTS_PET_DIR]

    # dictionary to contain different sets of data, e.g. all patients and only pet patients
    data_sets = {'all': pd.DataFrame(index=indices, columns=revasc_options, dtype='float64'),
                 'only pet': pd.DataFrame(index=indices, columns=revasc_options, dtype='float64'),
                 'predict revasc': pd.DataFrame(index=indices, columns=['all', 'only pet'], dtype='float64')}

    for dir in results_dirs:
        for file_name, data in iterate_results(dir):

            data_set = 'only pet' if 'only_pet' in file_name else 'all'
            result_index = int(file_name[-11]) if 'years' in file_name else max_time + int(include_unlimited_time)
            if 'early_revasc' not in file_name:
                df = data_sets[data_set]
                for revasc_option in revasc_options:
                    max_value = data.loc[:, (revasc_option, measure_by)].max()
                    if np.isnan(df.at[result_index, revasc_option]) or df.at[result_index, revasc_option] < max_value:
                        df.at[result_index, revasc_option] = max_value
            else:
                df = data_sets['predict revasc']
                max_value = data.loc[:, ('all', measure_by)].max()
                if np.isnan(df.at[result_index, data_set]) or df.at[result_index, data_set] < max_value:
                    df.at[result_index, data_set] = max_value

    for key, data in data_sets.items():

        plt.clf()
        fig, ax = plt.subplots()
        for col_name, vals in data.iteritems():
            vals = pd.Series(vals, dtype='float64').dropna()
            ax.plot(vals.index, vals, label=col_name)
            ax.scatter(vals.index, vals)
        plt.grid(True)
        ax.set_ylim((0.5, 1))
        ax.set_title(f'Test AUC scores for {key}')
        plt.legend()
        plt.savefig(os.getcwd() + f'\\Plots\\results_by_revasc_for_{key}', dpi=600)

# Plots 2 axes, on the left the results of the best model using PET-data and the best model using only CTA-data.
# On the right is plotted the difference (PET - CTA).
def plot_final_results_by_time():

    fixed_years = ((7, 8), (3, 5))

    for only_pet, (pet_fixed_year, no_pet_fixed_year) in zip([True, False], fixed_years):

        data_dir = os.getcwd() + f'\\Final test results{" only pet" if only_pet else ""}'
        fig, (ax, ax_diff) = plt.subplots(1, 2)
        fig.set_size_inches(12.8, 6.4)
        all_axes = [ax, ax_diff]
        data_pet, data_no_pet = None, None

        for file in os.listdir(data_dir):
            file_path = os.path.join(data_dir, file)
            if "not" in file_path:
                data_no_pet = pd.read_csv(file_path, index_col=0, header=[0,1])
                data_no_pet = data_no_pet.astype('float64')
            else:
                data_pet = pd.read_csv(file_path, index_col=0, header=[0, 1])
                data_pet = data_pet.astype('float64')

        data_pet = data_pet['all']
        data_no_pet = data_no_pet['all']

        times_str = [n for n in range(1, 9)]
        times_str.append("Unrestricted")
        times_ind = [n for n in range(1, 9)]
        times_ind.append(10)

        ax.plot(times_ind, data_pet['Test AUC'], label="Model using PET data")
        ax.scatter(times_ind, data_pet['Test AUC'])
        ax.plot(times_ind, data_no_pet['Test AUC'], label="Model not using PET data")
        ax.scatter(times_ind, data_no_pet['Test AUC'])
        ax_diff.plot(times_ind, data_pet['Test AUC'] - data_no_pet['Test AUC'], label="Test AUC", c='black')
        ax_diff.scatter(times_ind, data_pet['Test AUC'] - data_no_pet['Test AUC'], c='black')

        for axs in all_axes:
            axs.legend()
            axs.grid(True)
            axs.set_xticks(times_ind)
            axs.set_xticklabels(times_str)

        ax.set_ylim((0.4, 1))

        ax_diff.set_ylim((-0.2, 0.2))
        ax_diff.set_ylabel("Difference")

        ax.set_title(f"Results of the models")
        ax_diff.set_title("Added value of using PET data")

        fig.suptitle(f"Results for {'only the PET patients' if only_pet else 'all patients'}")

        plt.savefig(f'Plots\\Final test results for {"PET" if only_pet else "all"} patients', dpi=600)

    pass


def plot_PCA_component_event_ratios_by_time():

    settings = {'cta': True,
                'basic': True,
                'pet': True}

    cta = CTA_class.CTA_class(**settings)
    data = pd.read_csv("Training data.csv", index_col=0)
    data = cta.data_fill_na.loc[data.index, :]

    pca_mat = pd.read_csv("Result analysis\\pca_matrix_2_components.csv", index_col=0, header=0).drop(labels=["explained variance ratios"], axis=1)
    pca_mat = pca_mat.transpose()
    data = data.dot(pca_mat)
    data.columns = ['PCA1', 'PCA2']
    labels = pd.read_csv("Training labels.csv", index_col=0)

    pca2_values = data['PCA2']
    lower_q = pca2_values.quantile(0.25)
    upper_q = pca2_values.quantile(0.75)
    pca2_lower_quartile = pca2_values.loc[pca2_values <= lower_q]
    pca2_upper_quartile = pca2_values.loc[pca2_values >= upper_q]
    pca2_middle = pca2_values.loc[lower_q < pca2_values]
    pca2_middle = pca2_middle.loc[pca2_middle < upper_q]

    y_lower = labels.loc[pca2_lower_quartile.index, :]
    y_middle = labels.loc[pca2_middle.index, :]
    y_upper = labels.loc[pca2_upper_quartile.index, :]

    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches((6.4, 6.4))

    xs = list()
    y_ls = list()
    y_ms = list()
    y_us = list()
    for ind, col in enumerate(labels.columns):
        y_l = y_lower[col]
        y_m = y_middle[col]
        y_u = y_upper[col]
        year = ind
        if col == "event":
            continue

        xs.append(year)
        y_ls.append(y_l.sum()/labels[col].sum())
        y_ms.append(y_m.sum()/labels[col].sum()/2)
        y_us.append(y_u.sum()/labels[col].sum())

    col = "event"
    y_l = y_lower[col]
    y_m = y_middle[col]
    y_u = y_upper[col]
    year = 10

    xs.append(year)
    y_ls.append(y_l.sum() / labels[col].sum())
    y_ms.append(y_m.sum() / labels[col].sum()/2)
    y_us.append(y_u.sum() / labels[col].sum())


    ax.plot(xs, y_ls, label="First quartile")
    ax.scatter(xs, y_ls)
    ax.plot(xs, y_ms, label="Average of second and third quartile")
    ax.scatter(xs, y_ms)
    ax.plot(xs, y_us, label="Fourth quartile")
    ax.scatter(xs, y_us)

    ax.legend()

    ax.set_ylabel("Proportion of positive events")
    ax.set_xlabel("Observation year")
    ax.set_xticks([n for n in range(1, 11)])
    ax.set_xticklabels([n if n != 10 else "Unrestricted" for n in range(1, 11)])

    plt.show()

    pass


def plot_kaplan_meier():
    import kaplanmeier as km #TODO tähän täytyy viitata
    predictions_dir = os.getcwd() + '\\Predictions'
    label_file_names = {'Training': 'Training labels.csv',
              'Test': 'Test labels.csv'}
    time_file_path = os.getcwd() + '\\Data\\Processed Data\\Original data.csv'
    all_times = pd.read_csv(time_file_path, index_col=0, header=0)
    for file_name in iterate_files(predictions_dir):
        for group in ['Training', 'Test']:
            if group not in file_name:
                continue

            predictions = pd.read_csv(file_name, index_col=0, header=0)
            predictions = predictions['binary']
            labels = pd.read_csv(label_file_names[group], index_col=0, header=0)
            labels = labels.loc[predictions.index, 'event']
            times = all_times.loc[predictions.index, 'passed time']
            out = km.fit(times, labels, predictions)
            km.plot(out)
            plt.suptitle(file_name[49:])
            plt.savefig(f'Plots\\{file_name[49:]}.png', dpi=600)

    pass


def iterate_files(dir):
    for f in os.listdir(dir):
        if f.endswith(".csv"):
            yield dir + '\\' + f
    pass


if __name__ == "__main__":

    #plot_hist_PCA_values()
    #plot_scatter_PCA_values()
    #plot_results_by_time()
    #plot_results_grouped_by_revasc()
    #plot_final_results_by_time()
    #plot_PCA_component_event_ratios_by_time()
    plot_kaplan_meier()

    pass
