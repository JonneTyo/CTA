import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import CTA_main
import math


#returns a dic of all results with filename as keys and dataframe as items
def get_result_data(path='\\results'):

    to_return = dict()
    for f in os.listdir(os.getcwd() + path):
        if f.endswith(".csv"):
            df = pd.read_csv(os.getcwd() + path + '\\' + f, index_col=0)
            indx = pd.MultiIndex.from_arrays([df['model'], df['label']])
            df.index=indx
            df = df.loc[:, 'training_AUC':]
            to_return[f[:-4]] = df
    return to_return

#returns a dataframe with multi-index indices that contains data for each model-label pair
def get_best_models(dfs, measure_by='sensitivity', split_by=None):
    if split_by is None:
        split_by = 'all'
    to_return = dict()
    columns = ['Model Version', 'test_AUC', 'specificity', 'sensitivity', 'accuracy']
    indx1 = ['MI', 'CVD or MI', 'CVD or MI 2 Years', 'MI 4 Years']
    indx2 = [n for (n, _, _) in CTA_main.models]
    indx = pd.MultiIndex.from_product([indx1, indx2], names=['label', 'model'])
    sub_df = pd.DataFrame(index=indx, columns=columns)
    to_return[split_by] = sub_df
    if split_by != 'all':
        to_return['rest'] = sub_df.copy(deep=True)

    for key, df in dfs.items():
        if split_by in key or split_by == 'all':
            kw = split_by
        else:
            kw = 'rest'
        for (model, label), vals in df.iterrows():
            if model == 'SVM':
                continue
            old_val = to_return[kw].loc[(label, model), measure_by]
            if old_val < vals[measure_by] or math.isnan(old_val):
                for c in to_return[kw].columns:
                    if c == 'Model Version':
                        to_return[kw].loc[(label, model), c] = key[4:]
                    else:
                        to_return[kw].loc[(label, model), c] = vals[c]

    return to_return

# creates .csv files where each row has the results of the best data selection method
def get_best_results():
    all_results = get_result_data()
    split_by = 'only_PET'

    for m in ['sensitivity', 'test_AUC']:
        best_results = get_best_models(all_results, measure_by=m, split_by=split_by)
        for key, item in best_results.items():
            item.to_csv(os.getcwd() + f'\\result_analysis\\{key}_best_results_{m}.csv')
    print(best_results)
    pass


def plot_hist_passed_time_events(data, labels):
    x = data.loc[: , 'Passed time']/365
    x = x.astype(int)
    label_names = ['CVD', 'MI', 'CVD or MI']


    for l in range(labels.shape[1]):
        plt.clf()
        pos_lbls = list(labels.iloc[:, l] >= 1)
        x_pos = x.loc[pos_lbls]
        plt.hist(x_pos, rwidth=0.8, bins=[n for n in range(max(x_pos) + 1)])
        plt.xlabel('Years passed since start of follow-up')
        plt.title(f'{label_names[l]}, N={len(x_pos)}')
        plt.xticks(np.arange(0.1, (max(x_pos) + 1), step=1), labels=[n for n in range(max(x_pos) + 1)])
        plt.savefig(f'plots\\hist_passed_time_{label_names[l]}')
    pass




if __name__ == "__main__":
    pass
