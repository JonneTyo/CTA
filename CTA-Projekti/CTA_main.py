from feature_data import *
from data_handler import *
import cta_maths
import pandas as pd

from cta_maths import *
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt


def cta_main(time_analysis=True, only_PET=False, data_name="cta_data.csv", label_name="cta_labels.csv", gen_SIS=False, gen_SSS=False,
             drop_CTA=False, drop_PET=False):
    ############################################################################################
    ############################################################################################
    # SETTING UP THE DATA
    ############################################################################################
    ############################################################################################
    if seed is not None:
        iters = 1

    # feature_data variables
    global foi
    global ftr
    global orig_y_labels
    global feature_handling_frame
    global start_dates
    global end_dates
    global field_types

    y_labels = [n for n in orig_y_labels]

    # Set file path and rows to read
    csv_file_path = 'CTArekisteri_DATA_LABELS_2021-02-17_1146.csv'

    PECTUS_ids = csv_read('PECTUS.csv', index_col=None).drop('PVM_CTA', axis=1)

    # read data
    cta_data = csv_read(csv_file_path)
    cta_data.index = cta_data.index.map(str)
    cta_data.columns = csv_headers_original
    cta_data.drop(labels=[str(n) for n in list(range(2, 861))], inplace=True)




    # take only the desired rows
    # cta_data = cta_data.loc[0:data_amount]
    # cta_labels = cta_labels.loc[0:data_amount]

    # remove rows with all the values missing in the y_labels
    cta_data.dropna(how='all', subset=y_labels, inplace=True)

    # date handling. Turns  the dates into integers
    all_dates = [n for n in start_dates]
    all_dates.extend(end_dates)
    cta_data.loc[:, all_dates] = data_handle_dates(cta_data, start_dates, end_dates)

    # create feature "Passed time"
    data_passed_time(cta_data, start_dates, end_dates)

    # change strs to nums when possible
    cta_data = data_str_to_num(cta_data, cta_data_parser)
    PECTUS_ids = [str(n) for n in list(PECTUS_ids['ID'])  if n > 861]
    PECTUS_ids = cta_data.loc[PECTUS_ids]

    # has the data been checked?
    max_vals = np.max(cta_data.loc[:, ' (choice=cin (Teemu))':' (choice=gene (IidaK))'].T)
    max_vals = max_vals > 0
    cta_data = cta_data.loc[max_vals]
    # perform desired functions
    cta_data = data_desired_functions(cta_data, feature_handling_frame)
    PECTUS_ids = data_desired_functions(PECTUS_ids, feature_handling_frame, ignore=['Study indication'])

    cta_data = cta_data.append(PECTUS_ids, ignore_index=False)
    cta_data.drop_duplicates(inplace=True, ignore_index=False)

    # drop undesired columns and rows
    fois = cta_data[foi]
    for var in y_labels:
        fois[var] = cta_data.loc[:,var]
    cta_data.drop(labels=foi, axis='columns', inplace=True)
    cta_data = cta_data.drop(columns=ftr, errors='ignore')


    #if not only_PET:
    cta_data = cta_data.dropna(axis='columns', thresh=len(cta_data.index) - 500)

    for var in fois.columns:
        if var not in y_labels:
            cta_data[var] = fois[var]
    if not only_PET:
        cta_data = cta_data.dropna(axis='index', thresh=int((len(cta_data.columns) - 22)/2))
    if only_PET:
        cta_data.dropna(how='all', subset=[f'STR_seg_{n+1}' for n in range(17)], inplace=True)
    data_remove_indices(cta_data, fois)

    # assign cta_labels to only contain the labels
    cta_labels = fois.loc[:, y_labels]
    # remove the columns in ftr (features to remove)

    # Create SIS and SSS variables
    if not drop_CTA:
        n_classes=7
        cta_data = data_count_stenosis_types(cta_data, n_classes=n_classes)
        if gen_SIS:
            cta_data['SIS'] = cta_data.loc[:, 'Stenosis type 2 count':'Stenosis type 6 count'].sum(axis=1)
        if gen_SSS:
            cta_data['SSS'] = cta_data.loc[:, 'Stenosis type 4 count':'Stenosis type 6 count'].sum(axis=1)
        if gen_SIS or gen_SSS:
            cta_data.drop(labels=[f'Stenosis type {n} count' for n in range(1, 7)], inplace=True, axis=1)

    if drop_CTA or drop_PET:
        cta_data = data_drop(cta_data, drop_CTA, drop_PET)

    cta_data.fillna(value=0, inplace=True)
    cta_labels.fillna(value=0, inplace=True)

    # create label CV_DEATH or MI1
    new_label = (cta_labels[y_labels[0]] == 1) | (cta_labels[y_labels[1]] == 1)
    cta_labels.insert(cta_labels.shape[1], 'CVD or MI', new_label)
    y_labels.append('CVD or MI')
    #todo remove
    from cta_plots import plot_hist_passed_time_events
    plot_hist_passed_time_events(cta_data, cta_labels)
    #todo remove
    if time_analysis:
        new_label = new_label & (cta_data['Passed time'] <= 365*2)
        cta_labels.insert(cta_labels.shape[1], 'CVD or MI 2 Years', new_label)

        new_label = (cta_labels['MI1 - Confirmed'] == 1) & (cta_data['Passed time'] <= 365*4)
        cta_labels.insert(cta_labels.shape[1], 'MI 4 Years', new_label)


        cta_labels_time = cta_labels.loc[:, ['CVD or MI 2 Years','MI 4 Years']]
        cta_labels = cta_labels_time
        y_labels = ['CVD or MI 2 Years', 'MI 4 Years']
        global desired_label_names
        desired_label_names = y_labels
    cta_labels = cta_labels.astype(int)
    cta_data.drop(labels='Passed time', axis=1, inplace=True)



    # print information about data and labels
    print('Data entries: ', len(cta_data.index))
    print('Amount of features: ', len(cta_data.columns))
    print('Features:')
    print(cta_data.columns)
    print('\n Labels: ')
    print(cta_labels.columns)

    for lbl in y_labels:
        amount = cta_labels[lbl].sum()
        print(f'Cases of {lbl} : ', str(amount))

    cta_data.to_csv(data_name, index=False)
    cta_labels.to_csv(label_name, index=False)
    pass

def cta_train(cta_data, cta_labels, iters=100):
    cta_data = pd.read_csv(cta_data)
    cta_labels = pd.read_csv(cta_labels)
    # convert to numpy and split to train and testing
    cta_data = cta_data.to_numpy()
    cta_labels = cta_labels.to_numpy()
    train_ratio = 3/4


    ############################################################################################
    ############################################################################################
    # CONSTRUCTING MODELS
    ############################################################################################
    ############################################################################################

    to_return = pd.DataFrame(
        {'model': [], 'label': [], 'training_AUC': [], 'test_AUC': [], 'specificity': [], 'sensitivity': [], 'accuracy': []})

    for i in range(iters):
        print(f'iteration: {i + 1} / {iterations}')
        results = pd.DataFrame(
            {'model': [], 'label': [], 'training_AUC': [], 'test_AUC': [], 'specificity': [], 'sensitivity': [], 'accuracy': []})

        # split data

        x_train, x_test, y_train, y_test = data_train_test_split(cta_data, cta_labels, train_ratio=train_ratio, random_state=seed)
        lasso_coefs = list(range(y_train.shape[1]))
        linreg_coefs = list(range(y_train.shape[1]))

        for name, model, isClassifier in models:
            for l in range(int(not settings['time_analysis']), y_train.shape[1]):
                # Train and test AUC
                model = model.fit(x_train, y_train[:, l])
                pred_func = getattr(cta_maths, 'maths_' + name + '_pred')
                pred_train, pred_test = pred_func(model, x_train, x_test)
                train_auc = roc_auc_score(y_train[:, l], pred_train)
                test_auc = roc_auc_score(y_test[:, l], pred_test)
                fpr, sensitivity, thresholds = roc_curve(y_train[:, l], pred_train)
                specifity = 1 - fpr

                # Calculate youden's J value and create a new prediction on the test group based on the threshold
                youden_J = sensitivity + specifity - 1
                #youden_J_sens = (2 * sensitivity) + specifity - 1
                th = thresholds[np.argmax(youden_J)]
                #th_sens = thresholds[np.argmax(youden_J_sens)]

                pred_test_class = pred_test > th
                #pred_test_sens = pred_test > th_sens

                if not isClassifier:
                    accuracy = cta_maths.maths_bin_accuracy(pred_test_class, y_test[:, l])
                else:
                    accuracy = cta_maths.maths_bin_accuracy(pred_test, y_test[:, l])

                #spec_min, sens_min = cta_maths.maths_min_sens_weight(sensitivity, specifity, thresholds, y_test[:, l],
                #                                                     pred_test)

                # plot out different specs and sens for different weights on sensitivity

                # c_list = np.arange(1., 5.05, 0.05)
                # spec_list = list()
                # sens_list = list()
                # for c in c_list:
                #     youden_temp = c * sensitivity + specifity - 1
                #     th = thresholds[np.argmax(youden_temp)]
                #     pred_temp = pred_test > th
                #     sp, se = cta_maths.maths_spec_sens(y_test[:,l], pred_temp)
                #     spec_list.append(sp)
                #     sens_list.append(se)
                # plt.plot(c_list, spec_list, 'r', c_list, sens_list, 'b')
                # plt.axis([1,5, 0, 1])
                # plt.title('Model: ' + name + ' \n Label: ' + y_labels[l])
                # plt.show()


                # Specifity and sensitivity
                if not isClassifier:
                    specifity, sensitivity = cta_maths.maths_spec_sens(y_test[:, l], pred_test_class)
                else:
                    specifity, sensitivity = specifity[1], sensitivity[1]



                #spec_sens, sens_sens = cta_maths.maths_spec_sens(y_test[:, l], pred_test_sens)

                result = pd.DataFrame({
                    'model': name,
                    'label': desired_label_names[l],
                    'training_AUC': train_auc,
                    'test_AUC': test_auc,
                    'specificity': specifity,
                    'sensitivity': sensitivity,
                    'accuracy': accuracy
                }, index=[0])
                results = results.append(result, ignore_index=True)

                if name == 'Lasso':
                    if isinstance(lasso_coefs[l], int):
                        lasso_coefs[l] = model.coef_
                    else:
                        lasso_coefs[l] += model.coef_
                if name == 'LinReg':
                    if isinstance(linreg_coefs[l], int):
                        linreg_coefs[l] = model.coef_
                    else:
                        linreg_coefs[l] += model.coef_

        if to_return.shape[0] == 0:
            to_return = to_return.append(results, ignore_index=True)
        else:
            to_return.iloc[:, 2:] += results.iloc[:, 2:]




    to_return.iloc[:, 2:] = to_return.iloc[:, 2:]/iters
    print('train size: ', int(cta_data.shape[0] * train_ratio), ' test size: ', int(cta_data.shape[0]*(1-train_ratio)))
    '''


    lasso_frame = pd.DataFrame(data=lasso_coefs, index=y_labels, columns=features)
    #linreg_frame = pd.DataFrame(data=linreg_coefs, index=y_labels, columns=features)

    lasso_str = f'lasso_coefs_TIMED.csv' if time_analysis else f'lasso_coefs.csv'
    #linreg_str = f'linreg_coefs_TIMED.csv' if time_analysis else f'linreg_coefs.csv'

    lasso_frame = lasso_frame.T/iters
    #linreg_frame = linreg_frame.T/iters

    lasso_frame.to_csv(lasso_str)
    #linreg_frame.to_csv(linreg_str)
    for frame in [lasso_frame]:
        plt.pcolor(-frame, cmap='RdBu', vmin=-(np.abs(frame.to_numpy()).max()), vmax=np.abs(frame.to_numpy()).max())
        plt.yticks(np.arange(0.5, len(frame.index), 1), frame.index)
        plt.xticks(np.arange(0.5, len(frame.columns), 1), frame.columns)
        plt.show()
    '''

    return to_return


seed = None
#model name, model, isClassifier
models = [('Lasso', cta_maths.maths_lasso_model(random_state=seed), False),
          ('SVC', cta_maths.maths_SVC_model(), True)
          ]

'''

          ('boosted', cta_maths.maths_boosted_model(), False),
          ('LogReg', cta_maths.maths_loglasso_model(random_state=seed), False),
          ('RanFor', cta_maths.maths_random_forest_model(), False),
'''


def get_model_version(settings):
    to_return = ""
    for key, item in settings.items():
        if item:
            to_return = to_return + f'_{key}'
    return to_return

def iter_versions(settings, n_start=0):
    n_settings = 2**len(settings)

    for i in range(n_settings):
        if i < n_start:
            continue
        b_str = bin(i)[2:]
        if len(b_str) < len(settings):
            pad = len(settings) - len(b_str)
            for j in range(pad):
                b_str = '0' + b_str
        for j, (key,item) in zip(b_str, settings.items()):
            settings[key] = bool(int(j))
        if settings['drop_CTA'] and (settings['gen_SIS'] or settings['gen_SSS']):
            yield None
            continue
        yield settings



if __name__ == "__main__":
    pd.set_option('display.max_rows', 100)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.width', 0)
    gen_data_name = "data"
    gen_label_name = "labels"
    iterations = 100
    b_train = True
    b_gen_data = True

    settings = {
        'time_analysis': True, #restrict CVD or MI label to the cases where it took place within 2 years
                                #and MI label to the cases where it took place within 4 years
        'only_PET': False, #only choose the patients that have PET-results
        'drop_CTA' : True, #drop out CTA-related features
        'drop_PET' : False, #drop out PET-related features
        'gen_SIS' : False, #generate SIS-feature for each patient
        'gen_SSS' : False, #generate SSS-feature for each patient
        #'keras_train' : False, #train the keras model
        #'keras_pred' : False #predict with the trained keras model
    }

    run_all_versions = True

    def Main(settings):
        model_version_name = get_model_version(settings)
        gen_data_file = f'proc_data\\cta_{gen_data_name}{model_version_name}.csv'
        gen_label_file = f'proc_data\\cta_{gen_label_name}{model_version_name}.csv'
        gen_result_file = f'results\\cta_results{model_version_name}.csv'


        if b_gen_data:
            cta_main(time_analysis=settings['time_analysis'], only_PET=settings['only_PET'], data_name=gen_data_file, label_name=gen_label_file,
                    gen_SIS=settings['gen_SIS'], gen_SSS=settings['gen_SSS'], drop_CTA=settings['drop_CTA'], drop_PET=settings['drop_PET'])

        if b_train:
            results = cta_train(gen_data_file, gen_label_file, iters=iterations)
            results = results.round(decimals=3)
            results.sort_values(by=['label', 'model'], inplace=True)

            print(results)

            results.to_csv(gen_result_file)
        try:
            if settings['keras_pred']:
                y = 'CVD or MI'
                x_data, y_data, x_test, y_test = data_pp_keras(gen_data_name, gen_label_name, y=y) #todo
                if settings['keras_train']:

                    model = maths_keras_model(seed, name=y)
                    model, history = maths_keras_training(model, x_data, y_data, x_test, y_test)
                    model.save(f'D:\\Koulu\\CTA-Projekti\\Keras_models\\{y}')
                else:
                    model = keras.models.load_model(f'D:\\Koulu\\CTA-Projekti\\Keras_models\\{y}')
                    predictions = model.predict(x_test) #todo
        except KeyError:
            pass
        pass

    if run_all_versions:
        n_start = 0
        ver_counter = n_start + 1
        ver_amount = 2**len(settings)

        for ver in iter_versions(settings, n_start=n_start):
            print(f'Starting verion num. {ver_counter} out of {ver_amount}.')
            ver_counter += 1
            if ver is None:
                print('Skipping current version due to degeneracy.')
                continue
            print(f'Current version: ')
            for key, item in ver.items():
                if item:
                    print(key)
            Main(ver)


    else:
        Main(settings=settings)


