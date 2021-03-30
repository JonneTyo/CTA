from csv_handler_old import *
from data_handler_old import *
from feature_data import *
import cta_maths


def cta_main(model_list):
    ############################################################################################
    ############################################################################################
    # SETTING UP THE DATA
    ############################################################################################
    ############################################################################################

    # feature_data variables
    global foi
    global ftr
    global y_labels

    # read file
    csv_headers, cta_file = csv_read('CTArekisteri_DATA_2020-03-24_1051.csv')
    hyva_data = 2290
    # todo read all data once data is complete
    cta_file = data_lines(cta_file, range(hyva_data))
    cta_labels = data_columns(cta_file, data_find_feature_cols(csv_headers_original[1], y_labels))


    ############################################################################################
    ############################################################################################
    # COLLECTION OF DESIRED FEATURES (COLUMNS)
    ############################################################################################
    ############################################################################################


    # Create list with features of interest
    foi_list = data_columns(cta_file, data_find_feature_cols(csv_headers_original[1], foi))
    amount = 1500
    f_index = 0

    # Check the features have the correct amount of entries
    if isinstance(foi_list[0], list):
        for f in foi_list:
            if (len(f)-1 - cta_maths.maths_sum_Nones(f)) < amount:
                foi_list.pop(f_index)
                f_index -= 1
            f_index += 1

    # modify the foi_list to be just a list of column numbers
    ignore_list = []
    if isinstance(foi_list[0], list):
        for i in foi_list:
            ignore_list.append(i[0])
    else:
        ignore_list.append(foi_list[0])

    # General parsing of the data
    data_parse(cta_file, q=0.1, ignore=ignore_list)

    # updating foi
    for col in cta_file[0]:
        foi.append(csv_headers_original[1][col])
    foi = list(set(foi))

    for f in ftr:
        if f in foi:
            foi.pop(foi.index(f))

    # updating cta_file
    ftr_list = data_find_feature_cols(csv_headers_original[1], ftr)
    cta_file = data_pop_columns(cta_file, ftr_list)

    # removing the 'Complete?' fields
    ftr_list = []
    for i in cta_file[0]:
        if csv_headers_original[1][i] == 'Complete?':
            ftr_list.append(i)
    cta_file = data_pop_columns(cta_file, ftr_list)

    ############################################################################################
    ############################################################################################
    # ELIMINATING POOR DATA ENTRIES (ROWS)
    ############################################################################################
    ############################################################################################

    tolerance = 10

    # eliminating rows that's missing too much data
    final_data = []
    final_labels = []
    if isinstance(cta_labels[0], list):
        for i, j in zip(cta_file, range(len(cta_labels[0]))):
            if cta_maths.maths_sum_Nones(i) < tolerance:
                temp_list = list()
                final_data.append(i)
                for k in range(len(cta_labels)):
                    temp_list.append(cta_labels[k][j])
                final_labels.append(temp_list)
    else:
        for i, j in zip(cta_file, cta_labels):
            if cta_maths.maths_sum_Nones(i) < tolerance:
                final_data.append(i)
                final_labels.append(j)
    # constructing a list of bools representing wether a row should be kept or not
    keep_list = []
    if isinstance(final_labels[0], list):
        for i in range(len(final_labels)):
            value_found = False
            for j in range(len(final_labels[0])):
                if final_labels[i][j] is not None:
                    value_found = True
                    break
            if value_found:
                keep_list.append(True)
            else:
                keep_list.append(False)
    else:
        for i in range(len(final_labels)):
            if final_labels[i] is None:
                keep_list.append(False)
            else:
                keep_list.append(True)

    # constructing temp lists
    final_data2 = []
    final_labels2 = []
    counter = 0
    for i in keep_list:
        if i:
            counter += 1
    print(counter)
    if isinstance(final_labels[0], list):
        for i in range(len(keep_list)):
            if keep_list[i]:
                row_list = list()
                for j in range(len(final_labels[0])):
                    row_list.append(final_labels[i][j])
                final_data2.append(final_data[i])
                final_labels2.append(row_list)
    else:
        for i in range(len(keep_list)):
            if keep_list[i]:
                final_data2.append(final_data[i])
                final_labels2.append(final_labels[i])

    final_data_headers = []
    for i in final_data[0]:
        final_data_headers.append(csv_headers_original[1][i])
    csv_write('final_data_headers.csv', final_data_headers)
    final_data = final_data2[1:]
    csv_write('final_data.csv', final_data, dict=final_data_headers)
    final_labels = final_labels2[1:]
    csv_write('final_labels.csv', final_labels)


    ############################################################################################
    ############################################################################################
    # CONSTRUCTING MODELS
    ############################################################################################
    ############################################################################################
    final_data, final_labels = data_as_matrix(final_data), data_as_matrix(final_labels)
    data_convert_nans(final_data, 0)
    data_convert_nans(final_labels, 0)
    folds = data_kfold(final_data, n=5)
    print('Data entries: ', len(final_data))
    print('Amount of features: ', len(final_data[0]))
    print('Features: ')
    print(final_data_headers)
    random_seed = None

    if 'Keras' in model_list:
        print('################################################################')
        print('################################################################')
        print('KERAS DEEP LEARN MODEL ')
        print('################################################################')
        print('################################################################')

        evals = []
        for fold in folds:
            model = maths_keras_seqmodel((128, 128, 32), 1, th=0.1)
            maths_keras_fit(model, final_data[fold[0], :], final_labels[fold[0], :], epochs=10)
            evals.append(model.evaluate(final_data[fold[1], :], final_labels[fold[1], :]))

        mean_accuracy = 0
        mean_loss = 0

        for eval in evals:
            mean_accuracy += eval[1]
            mean_loss += eval[0]

        mean_accuracy = mean_accuracy/len(evals)
        mean_loss = mean_loss/len(evals)

        print('Mean accuracy: ', mean_accuracy, ' Mean loss: ', mean_loss)

    if 'LinReg' in model_list:
        print('################################################################')
        print('################################################################')
        print('LINEAR REGRESSION MODEL ')
        print('################################################################')
        print('################################################################')

        model = maths_linear_reg_model(random_state=random_seed)
        evals = []
        for fold in folds:

            x_train = final_data[fold[0], :]
            y_train = final_labels[fold[0], :]
            x_test = final_data[fold[1], :]
            y_test = final_labels[fold[1], :]
            for i in range(y_test.shape[1]):
                model = model.fit(x_train, y_train[:,i])
                train_pred = model.predict(x_train)
                test_pred = model.predict(x_test)
                m = tf.keras.metrics.BinaryAccuracy()
                m.update_state(y_train[:,i], train_pred)
                train_err = m.result().numpy()

                m = tf.keras.metrics.BinaryAccuracy()
                m.update_state(y_test[:,i], test_pred)
                val_err = m.result().numpy()
                from sklearn.metrics import roc_auc_score
                evaluation = list()
                evaluation.append(train_err)
                evaluation.append(val_err)
                print(y_test[:,i].shape, test_pred.shape)
                auc = roc_auc_score(y_test[:,i], test_pred)
                evaluation.append(auc)

        print('Train err, ', 'Val err, ', 'AUC, ')
        for e in evals:
            print(e)

        if "Randfor" in model_list:

            print('################################################################')
            print('################################################################')
            print('Random Forest ')
            print('################################################################')
            print('################################################################')

            model = maths_random_forest_model(depth=4,random_state=random_seed)
            evals = []
            for fold in folds:
                x_train = final_data[fold[0], :]
                y_train = final_labels[fold[0], :]
                y_train.ravel()
                x_test = final_data[fold[1], :]
                y_test = final_labels[fold[1], :]
                y_test.ravel()

                model.fit(x_train, y_train)
                train_pred = model.predict(x_train)
                test_pred = model.predict(x_test)
                m = tf.keras.metrics.BinaryAccuracy()
                m.update_state(y_train, train_pred)
                train_err = m.result().numpy()

                m = tf.keras.metrics.BinaryAccuracy()
                m.update_state(y_test, test_pred)
                val_err = m.result().numpy()

                from sklearn.metrics import auc
                from sklearn.metrics import roc_curve

                fpr, tpr, thresholds = roc_curve(y_test, test_pred, pos_label=1)

                evals.append((train_err, val_err, auc(fpr, tpr), tpr, fpr))

            print('Train err, ', 'Val err, ', 'AUC, ', 'tpr, ', 'fpr')
            for e in evals:
                print(e)








    pass


models = ['LinReg']
cta_main(models)
