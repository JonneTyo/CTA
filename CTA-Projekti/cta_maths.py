import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import Ridge
from sklearn.linear_model import ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.svm import SVR
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def linear_map(new_range, old_range):
    def wrapper_func(x):
        return (x - old_range[0])/(old_range[1] - old_range[0])*(new_range[1] - new_range[0]) + new_range[0]
    return wrapper_func

def maths_mean_squared_err(labels,pred):
    return mean_squared_error(y_true=labels, y_pred=pred)

def maths_lasso_model(random_state=None, alpha=0.01):
    return Lasso(random_state=random_state, alpha=alpha, positive=False)

def maths_Lasso_pred(fitted_model, x_train, x_test):
    return fitted_model.predict(x_train), fitted_model.predict(x_test)

def maths_LogReg_model(random_state=None, solver='saga'):
    return LogisticRegression(penalty='elasticnet', random_state=random_state, solver=solver, max_iter=10000,
                              fit_intercept=True, l1_ratio=0.2, class_weight='balanced')

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

def maths_SVR10_model(**kwargs):
    return SVR(kernel='linear', gamma='scale', C=1)

def maths_SVR05_model(**kwargs):
    return SVR(kernel='linear', gamma='scale', C=0.5)

def maths_SVR03_model(**kwargs):
    return SVR(kernel='linear', gamma='scale', C=0.3)

def maths_ridge_model(**kwargs):
    return Ridge(alpha=0.5)

def maths_ElasticNet_model(**kwargs):
    return ElasticNet(alpha=0.5)

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

def maths_keras_old_model(random_state=None, weights_load_path=None):
    n_input = 64

    inputs = keras.Input((n_input,1))
    conv1 = layers.Conv1D(2, 4, activation='relu', kernel_initializer='random_normal', bias_initializer='zeros',
                          input_shape=inputs.shape, padding='same')(inputs)
    conv2 = layers.Conv1D(4, 2, strides=2, activation='relu', kernel_initializer='random_normal',
                          bias_initializer='random_normal')(conv1)
    pool1 = layers.MaxPool1D(pool_size=2, strides=2)(conv1)
    conv3 = layers.Conv1D(8, 2, strides=2, activation='relu', kernel_initializer='random_normal',
                          bias_initializer='random_normal')(pool1)
    merge1 = layers.concatenate([pool1, conv2], axis=2)
    pool2 = layers.MaxPool1D(pool_size=2, strides=2)(merge1)
    conv4 = layers.Conv1D(16, 2, strides=2, activation='relu', kernel_initializer='random_normal',
                          bias_initializer='random_normal')(pool2)
    merge2 = layers.concatenate([pool2, conv3], axis=2)
    pool3 = layers.MaxPool1D(pool_size=2, strides=2)(merge2)
    conv5 = layers.Conv1D(32, 2, strides=2, activation='relu', kernel_initializer='random_normal',
                          bias_initializer='random_normal')(pool3)

    merge3 = layers.concatenate([pool3, conv4], axis=2)
    pool4 = layers.MaxPool1D(pool_size=2, strides=2)(merge3)
    merge4 = layers.concatenate([pool4, conv5], axis=2)
    dense = layers.Dense(1, activation='sigmoid')(merge4)
    pool5 = layers.MaxPool1D(pool_size=4, strides=1)(dense)
    model = keras.Model(inputs=inputs, outputs=pool5)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])

    if weights_load_path:
        print(f'LOADING WEIGHTS AT PATH:  {weights_load_path}')
        model.load_weights(weights_load_path)

    return model

def maths_keras_model(random_state=None, weights_load_path=None):
    inputs = keras.Input((17, 1))
    dens1 = layers.Dense(17, bias_initializer='normal')(inputs)
    merge1 = layers.concatenate([inputs, dens1], axis=2)
    conv1 = layers.Conv1D(6, 2, strides=1, bias_initializer='normal')(merge1)
    conv2 = layers.Conv1D(6, 3, strides=1, bias_initializer='normal')(merge1)
    pad2 = layers.ZeroPadding1D((0, 1))(conv2)
    conv3 = layers.Conv1D(6, 4, strides=1, bias_initializer='normal')(merge1)
    pad3 = layers.ZeroPadding1D((0, 2))(conv3)
    conv4 = layers.Conv1D(6, 5, strides=1, bias_initializer='normal')(merge1)
    pad4 = layers.ZeroPadding1D((0, 3))(conv4)
    merge2 = layers.concatenate([conv1, pad2, pad3, pad4], axis=2)
    masked1 = layers.Masking()(merge2)
    dens2 = layers.Dense(4, activation='relu', bias_initializer='normal')(masked1)
    pool1 = layers.MaxPool1D(pool_size=16)(dens2)
    dens3 = layers.Dense(1, activation='sigmoid')(pool1)
    model = keras.Model(inputs=inputs, outputs=dens3)
    model.compile(optimizer=keras.optimizers.Adam(lr=1e-3), loss=tf.keras.losses.binary_crossentropy,
                  metrics=[tf.keras.metrics.BinaryAccuracy(), tf.keras.metrics.AUC()])

    if weights_load_path:
        print(f'LOADING WEIGHTS AT PATH:  {weights_load_path}')
        model.load_weights(weights_load_path)

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










