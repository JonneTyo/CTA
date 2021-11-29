from CTA_class import CTA_class
from CTA_class import iter_options
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import kruskal
import os


def get_similar_train_test_splits(data, labels, p_val_th=0.05):
    for i in range(1000):
        data_is_similar = True
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.25, stratify=labels['event'], random_state=i)

        # check if each variable has similar mean in training and test groups. If not (p_val <= p_val_th), generate again
        for (col_train, train_vals), (col_test, test_vals), in zip(X_train.iteritems(), X_test.iteritems()):
            if col_train != col_test:
                raise IndexError("DIFFERENT COLUMNS IN ITERATION")
            if train_vals.min() == train_vals.max() and test_vals.min() == test_vals.max():
                continue
            _, p_val = kruskal(train_vals, test_vals)
            if p_val <= p_val_th:
                data_is_similar = False
                break

        # same as above but for labels
        if data_is_similar:
            upper_limit = 1.15
            lower_limit = 1 / upper_limit
            for col, vals in labels.iteritems():
                train_vals = vals.loc[y_train.index]
                test_vals = vals.loc[y_test.index]
                train_proportion = train_vals.sum() / train_vals.shape[0]
                test_proportion = test_vals.sum() / test_vals.shape[0]
                proportion = train_proportion / test_proportion
                if lower_limit > proportion or proportion > upper_limit:
                    data_is_similar = False
                    break
        if data_is_similar:
            print(f'Random state: {i}')
            break
    else:
        raise ValueError("Could not create similar data splits")
    return X_train, X_test, y_train, y_test


setting_dict = {
    'pet': True,
    'cta': True,
    'basic': True,
    'pca': True,
    'ctapca': True,
}
events = [f'event {n} years' for n in range(1, 9)]
events = ['event'] + events

cta_pet = CTA_class(only_pet=True)
cta_no_pet = CTA_class()
all_data = cta_pet.original_data.loc[:,
           cta_pet.BASIC_VARIABLES + cta_pet.RISK_VARIABLES + cta_pet.CTA_VARIABLES + cta_pet.PET_VARIABLES + cta_pet.PCA_VARIABLES + cta_pet.PET_PCA_VARIABLES + cta_pet.CTA_PCA_VARIABLES + cta_pet.PET_CTA_PCA_VARIABLES].drop(
    labels=['passed time'], axis=1)

# all_data.fillna(fill_dict, inplace=True)

pet_indices = cta_pet.data.index
no_pet_indices = cta_no_pet.data.drop(labels=pet_indices).index

X_train_pet, X_test_pet, y_train_pet, y_test_pet = get_similar_train_test_splits(all_data.loc[pet_indices, :],
                                                                                 cta_pet.original_data.loc[pet_indices, events])
X_train_no_pet, X_test_no_pet, y_train_no_pet, y_test_no_pet = get_similar_train_test_splits(all_data.loc[no_pet_indices, :],
                                                                                             cta_no_pet.original_data.loc[no_pet_indices, events])


X_train = pd.concat([X_train_no_pet, X_train_pet])
X_test = pd.concat([X_test_no_pet, X_test_pet])
y_train = pd.concat([y_train_no_pet, y_train_pet])
y_test = pd.concat([y_test_no_pet, y_test_pet])

fill_dict = {n: X_train[n].median() for n in X_train.columns}
for col in cta_pet.PET_VARIABLES:
    fill_dict[col] = X_train_pet[col].median()

X_train.to_csv("Training data.csv")
y_train.to_csv("Training labels.csv")
X_test.to_csv("Test data.csv")
y_test.to_csv("Test labels.csv")

X_train.fillna(fill_dict, inplace=True)
X_train_pet.fillna(fill_dict, inplace=True)
X_test.fillna(fill_dict, inplace=True)
X_test_pet.fillna(fill_dict, inplace=True)
X_train.to_csv("Training data filled.csv")
X_train_pet.to_csv("PET Training data filled.csv")
X_test.to_csv("Test data filled.csv")
X_test_pet.to_csv("PET Test data filled.csv")

curr_opt_count = 1
obs_years = [1, 2, 3, 4, 5, 6, 7, 8, None]

all_patients_str = "all patients"
pet_patients_str = "selected patients"
using_pet_data_str = "using PET data"
not_using_pet_data_str = "not using PET data"

final_models = {all_patients_str + ' ' + using_pet_data_str:
                    (3,
                     {'pet': True,
                      'cta': True,
                      'basic': True,
                      'pca': True},
                     'ridge'),
                all_patients_str + ' ' + not_using_pet_data_str: (5, {
                    'cta': True,
                    'basic': True,
                    'ctapca': True
                }, 'linreg'),
                pet_patients_str + ' ' + using_pet_data_str: (7, {
                    'only_pet': True,
                    'pet': True,
                    'cta': True,
                    'basic': True,
                    'pca': True
                }, 'linreg'),
                pet_patients_str + ' ' + not_using_pet_data_str: (8, {
                    'only_pet': True,
                    'cta': True,
                    'basic': True,
                    'ctapca': True
                }, 'linreg')}


# Takes in two dictionaries with same keys. model_predictions must contain iterables while bin_ths must contain floats.
# Returns a new dictionary with same keys where the values of model_predictions have been turned into binary values
# based on whether or not they are larger than the corresponding bin_ths value.
def float_to_binary_pred(model_predictions, bin_ths):
    to_return = dict()
    for model_name in model_predictions.keys():
        pred = model_predictions[model_name]
        th = bin_ths[model_name]
        bin_pred = [int(p >= th) for p in pred]
        to_return[model_name] = bin_pred

    return to_return


for model_opt, (fixed_year, opt, model_name) in final_models.items():



    only_pet = pet_patients_str in model_opt
    event_str = f'event {fixed_year} years'
    cta = CTA_class(**opt)
    print(f"Model option: {model_opt}")
    print(f'Settings to string: {cta.settings_to_str}')
    print(f'Fixed year: {fixed_year} \n')
    cta.label = event_str
    cta.X_train, cta.X_test, cta.y_train, cta.y_test = X_train.loc[:, cta.data.columns], X_test.loc[:, cta.data.columns], y_train.loc[:, event_str].astype(int), y_test.loc[:, event_str].astype(int)
    if only_pet:
        cta.X_train, cta.X_test, cta.y_train, cta.y_test = X_train_pet.loc[:, cta.data.columns], X_test_pet.loc[:, cta.data.columns], y_train_pet.loc[:, event_str].astype(int), y_test_pet.loc[:, event_str].astype(int)

    categories = ['value', 'binary']
    training_predictions = pd.DataFrame(index=cta.X_train.index, columns=categories)
    test_predictions = pd.DataFrame(index=cta.X_test.index, columns=categories)
    cta.train_models()
    cta.predict(cta.X_train)
    bin_ths = cta.binary_threshold
    training_predictions['value'] = cta.model_predictions[model_name]
    training_predictions['binary'] = float_to_binary_pred(cta.model_predictions, bin_ths)[model_name]
    cta.predict(cta.X_test)
    test_predictions['value'] = cta.model_predictions[model_name]
    test_predictions['binary'] = float_to_binary_pred(cta.model_predictions, bin_ths)[model_name]

    training_predictions.to_csv(f"Predictions\\Training predictions {cta.settings_to_str}.csv")
    test_predictions.to_csv(f'Predictions\\Test predictions {cta.settings_to_str}.csv')


    cta.model_predictions = float_to_binary_pred(cta.model_predictions, bin_ths)
    results = pd.DataFrame(index=[n if n else 'Unrestricted' for n in obs_years], columns=cta.results.columns)
    for obs_year in obs_years:

        obs_year_str = f'event {obs_year} years' if obs_year else f'event'


        obs_year_ind = obs_year if obs_year else "Unrestricted"
        cta.label = obs_year_str
        cta.y_test = y_test.loc[:, obs_year_str].astype(int) if not only_pet else y_test_pet.loc[:, obs_year_str].astype(int)
        #cta.y_train = y_train.loc[:, obs_year_str].astype(int) if not only_pet else y_train_pet.loc[:, obs_year_str].astype(int)
        results.loc[obs_year_ind, :] = cta.results.loc[model_name, :]
        #print(cta.results)

    results.to_csv(os.getcwd() + f'\\Final test results{" only pet" if only_pet else ""}\\{model_opt}.csv')


    # print(f"Starting option no. {curr_opt_count}")
    # cta = CTA_class(**opt)
    #
    # cta.X_train, cta.X_test, cta.y_train = X_train.loc[:, cta.data.columns], X_test.loc[:, cta.data.columns], y_train.loc[:,
    #                                                                                                           fixed_year_str].astype(int)
    # cta.label = fixed_year_str
    # cta.y_test = y_test.loc[:, cta.label].astype(int)
    # # cta.train_models()
    # # cta.predict(cta.X_train)
    # cta.predict(cta.X_test)
    # bin_ths = cta.binary_threshold
    # cta.model_predictions = float_to_binary_pred(cta.model_predictions, bin_ths)
    # for event in events:
    #     cta.y_train = y_train.loc[:, event].astype(int)
    #     cta.label = event
    #
    #     # cta.train_models()
    #     # cta.predict(cta.X_train)
    #     # train_results = cta.training_results
    #     # train_results.to_csv(os.getcwd() + f'\\Final results\\{cta.settings_to_str}_TRAINING_{event}.csv')
    #     # cta.y_test = y_test.loc[:, event].astype(int)
    #     # cta.predict(cta.X_test)
    #     # cta.model_predictions = float_to_binary_pred(cta.model_predictions, bin_ths)
    #
    #     results = cta.training_results
    #     results.to_csv(os.getcwd() + f'\\Final results\\{cta.settings_to_str}_{event}_fixed_years_{fixed_year}.csv')
    #
    # print("PET version")
    # cta_pet = CTA_class(only_pet=True, **opt)
    # cta_pet.X_train, cta_pet.X_test, cta_pet.y_train = X_train_pet.loc[:, cta_pet.data.columns], X_test_pet.loc[:,
    #                                                                                              cta_pet.data.columns], y_train_pet.loc[:,
    #                                                                                                                     fixed_year_str].astype(
    #     int)
    # cta_pet.label = fixed_year_str
    # cta_pet.y_test = y_test_pet.loc[:, cta_pet.label].astype(int)
    # cta_pet.train_models()
    # # cta_pet.predict(cta_pet.X_train)
    # cta_pet.predict(cta_pet.X_test)
    # bin_ths_pet = cta_pet.binary_threshold
    # cta_pet.model_predictions = float_to_binary_pred(cta_pet.model_predictions, bin_ths_pet)
    # for event in events:
    #     cta_pet.y_train = y_train_pet.loc[:, event].astype(int)
    #
    #     cta_pet.label = event
    #
    #     # cta_pet.train_models()
    #     # cta_pet.predict(cta_pet.X_train)
    #     # train_results = cta_pet.training_results
    #     # train_results.to_csv(os.getcwd() + f'\\Final results only pet\\{cta_pet.settings_to_str}_TRAINING_{event}.csv')
    #
    #     # cta_pet.y_test = y_test_pet.loc[:, event].astype(int)
    #     # cta_pet.predict(cta_pet.X_test)
    #     # cta_pet.model_predictions = float_to_binary_pred(cta_pet.model_predictions, bin_ths_pet)
    #
    #     results = cta_pet.training_results
    #     results.to_csv(os.getcwd() + f'\\Final results only pet\\{cta_pet.settings_to_str}_{event}_fixed_years_{fixed_year}.csv')
    #
    # curr_opt_count += 1
