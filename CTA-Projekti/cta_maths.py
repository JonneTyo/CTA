import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

#Sums over a list and returns the number of Nones
def maths_sum_Nones(list):
    n = 0
    for i in list:
        if i is None:
            n += 1
    return n


#todo PCA
def maths_PCA(x, labels, n_comps=2, max_i=100):

    pass


def maths_mean_squared_err(labels,pred):
    return mean_squared_error(y_true=labels, y_pred=pred)

def maths_lasso_model(random_state=None, alpha=0.01):
    return Lasso(random_state=random_state, alpha=alpha)

def maths_Lasso_pred(fitted_model, x_train, x_test):
    return fitted_model.predict(x_train), fitted_model.predict(x_test)

def maths_loglasso_model(random_state=None, solver='saga'):
    return LogisticRegression(penalty='elasticnet', random_state=random_state, solver=solver, max_iter=10000,
                              fit_intercept=True, l1_ratio=0.5)

def maths_LogReg_pred(fitted_model, x_train, x_test):

    pred_train = list()
    pred_test = list()
    for tr in fitted_model.predict_proba(x_train):
        pred_train.append(tr[1])

    for te in fitted_model.predict_proba(x_test):
        pred_test.append(te[1])

    return pred_train, pred_test

def maths_linreg_model(random_state=None):
    return LinearRegression(normalize=True)

def maths_LinReg_pred(fitted_model, x_train, x_test):
    return fitted_model.predict(x_train), fitted_model.predict(x_test)

def maths_random_forest_model(depth=None,random_state=None):

    return RandomForestRegressor(n_estimators=10, max_depth=depth, random_state=random_state, max_features='sqrt')

def maths_RanFor_pred(fitted_model, x_train, x_test):
    return fitted_model.predict(x_train), fitted_model.predict(x_test)

def maths_spec_sens(y_true, y_pred, pos_label=1):

    tp = fp = tn = fn = 0

    for i, j in zip(y_true, y_pred):

        if i == pos_label:
            if j == pos_label:
                tp += 1
            else:
                fn += 1
        else:
            if j == pos_label:
                fp += 1
            else:
                tn += 1

    return tn/(tn + fp), tp/(tp + fn)

def maths_SVC_model(**kwargs):
    return SVC(gamma='scale', class_weight='balanced')

def maths_SVC_pred(fitted_model, x_train, x_test):
    return fitted_model.predict(x_train), fitted_model.predict(x_test)

def maths_SVR_model(**kwargs):
    return SVR(gamma='scale')

def maths_SVR_pred(fitted_model, x_train, x_test):
    return fitted_model.predict(x_train), fitted_model.predict(x_test)

def maths_boosted_model(random_state=None):
    return GradientBoostingClassifier(random_state=random_state, max_features='auto')

def maths_boosted_pred(fitted_model, x_train, x_test):
    pred_train = list()
    pred_test = list()
    for tr in fitted_model.predict_proba(x_train):
        pred_train.append(tr[1])

    for te in fitted_model.predict_proba(x_test):
        pred_test.append(te[1])

    return pred_train, pred_test

def maths_min_sens_weight(sens, spec, thresholds, true, pred, step=0.01, upper=5):
    c_list = np.arange(1, upper + step, step)

    for c in c_list:
        y = c*sens + spec - 1
        th = thresholds[np.argmax(y)]
        pred_class = pred > th
        sp, se = maths_spec_sens(true, pred_class)
        if se > sp:
            return sp, se
        if c == 1:
            sp1 = sp
            se1 = se
    else:
        return sp1, se1

def maths_bin_accuracy(pred, true):
    n = min(len(pred), len(true))
    c = 0
    assert n > 0, 'both arguments pred and true must be non-empty list-like objects'
    for p, t in zip(pred, true):
        c += int(p == t)

    return c/n

def maths_keras_model(random_state=None, name=None):
    model = keras.Sequential(
        [
            keras.Input(shape=(51,)),
            layers.Dense(51, activation="relu", kernel_initializer='random_normal', bias_initializer='random_normal'),
            layers.Dense(14, activation="relu", kernel_initializer='random_normal', bias_initializer='random_normal'),
            layers.Dense(2, activation="relu", kernel_initializer='random_normal', bias_initializer='random_normal')

        ],
        name=f'keras_{name}'
    )
    return model

def maths_keras_training(model, x_train, y_train, x_test, y_test, batch_size=64, epochs=10):
    model.compile(
        loss='categorical_crossentropy',
        optimizer = 'adam'
    )
    history = model.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
    )
    return model, history

if __name__ == "__main__":

    x = tf.ones((1,4))
    print(x)










