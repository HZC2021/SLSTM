from math import sqrt
from numpy import concatenate
from matplotlib import pyplot
from pandas import read_csv
from sklearn.preprocessing import MinMaxScaler
import joblib
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score
from keras.models import Sequential
from keras.layers import Dense, Activation  # , RNN
from mymodel1 import SLSTMCell1
from mymodel2 import SLSTMCell2
import tensorflow as tf
import os
from keras.layers import LSTM
import numpy as np
from pickle import dump
from keras.regularizers import L1
import math
from ordinarylstm import CustomLSTMCell
from kerasrnn import RNN
import random
import datetime

# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    x = np.empty(shape = (0,8,1))
    y = np.empty(shape = (0,8))
    for i in range(len(data) // 8 - 1):
        x = np.concatenate((x, [data[8*i:8*(i+1),:]]),axis = 0)
        y = np.concatenate((y, [data[8*(i+1):8*(i+2), 0]]), axis = 0)
    return x,y

def scheduler(epoch, lr):
    if epoch < 50:
        return lr
    else:
        return lr * 1 / (1 + 5e-5 * epoch)

def create_model(SLSTM=True, units=50, drop_out=0., l1=1e-3, batch_input_shape=None):
    model = Sequential()
    if SLSTM:
        model.add(RNN(SLSTMCell1(units, dropout=drop_out, recurrent_dropout=drop_out,
                                     kernel_regularizer=L1(l1=l1)),
                      SLSTMCell2(units, dropout=drop_out, recurrent_dropout=drop_out,
                                kernel_regularizer=L1(l1=l1)),
                      CustomLSTMCell(units, dropout=drop_out, recurrent_dropout=drop_out,
                                kernel_regularizer=L1(l1=l1)),
                      unroll=True, #return_sequences=True,
                      stateful=True))
    else:
        model.add(LSTM(units, dropout=drop_out, recurrent_dropout=drop_out,
                       kernel_regularizer=L1(l1=l1), stateful=True))

    model.add(Activation('relu'))
    model.add(Dense(8, kernel_regularizer=L1(l1=l1)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-2), loss='mse')
    return model


def trainwithval(trainset, valset, units=50, drop_out=0., batch_size=32, epochs=50, L1_value=0., L1_name="0"):
    # design network
    model = create_model(SLSTM=True, units=units, drop_out=drop_out, l1=L1_value, )
    # define callback func
    file_name = "{epoch:02d}_{loss:.6f}.ckpt"
    checkpoint_path = os.path.join('./std_record_aci_aru03/train_cmba_%s_aru%02d_SLSTM_L%s_u%d_dp%.2f_std_t0' % (name, aru, L1_name, units, drop_out), file_name)
    callback_lr = tf.keras.callbacks.LearningRateScheduler(scheduler)
    callback_savemodel = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        verbose=0,
        save_weights_only=True,
        save_freq="epoch")
    # devide data for training and validation
    train_X, train_y = trainset
    val_X, val_y = valset
    # fit network
    val_epoch = []
    for i in range(epochs):
        history = model.fit(train_X, train_y, epochs=1, batch_size=batch_size,  # validation_data=(val_X, val_y),
                            callbacks=[callback_lr, callback_savemodel], verbose=2,
                            shuffle=False)
        model.reset_states()
        val_len = len(val_y)
        # train_len = len(train_y)
        run_X = np.concatenate((train_X, val_X), axis=0)
        # run_y = np.concatenate((train_y, val_y), axis=0)
        run_yhat = model.predict(run_X, batch_size=batch_size)
        tmp = mean_squared_error(run_yhat[-val_len:, :], val_y[:, :])
        # tmp = mean_squared_error(run_yhat[0:-val_len, :], train_y[:, :])
        print("epoch:%d val mse:" % i, tmp)
        val_epoch.append([i, tmp, history.history['loss']])
        model.reset_states()
    np.save(r"std_aci_aru03\val_epoch_cmba_%s_aru%02d_L%s_u%d_dp%.2f_t0.npy" % (name, aru, L1_name, units, drop_out), val_epoch)

    return model




if __name__ == "__main__":
    # load dataset
    name = "aci"
    aru = 3
    data = np.load(r"cbma\%s_aru%02d.npy" % (name, aru), allow_pickle=True)
    values = data[:, 1]
    L1_v = 1e-6
    L1_array = "1e-6"

    scaled = values[:, np.newaxis].astype('float32')
    n_hours = 8
    n_features = 1
    # frame as supervised learning
    X, y = series_to_supervised(scaled, n_hours, 1)
    num = len(y)
    train_num = round(num * 0.6)
    val_num = round(num * 0.2)
    test_num = num - train_num - val_num
    train_X = X[:train_num, :, :]
    train_y = y[:train_num, :]
    train_val_X = X[:train_num + val_num, :, :]
    train_val_y = y[:train_num + val_num, :]
    u = np.mean(train_val_X[:,:,0], dtype=np.float32)
    std = np.std(train_val_X[:,:,0], dtype=np.float32)
    np.save("is_u.npy", u)
    np.save("is_std.npy", std)

    val_X = X[train_num:train_num + val_num, :, :]
    val_y = y[train_num:train_num + val_num, :]
    test_X = X[train_num + val_num:, :, :]
    test_y = y[train_num + val_num:, :]

    for i in range(len(train_X)):
        for j in range(8):
            train_X[i, j, 0] = (train_X[i, j, 0] - u) / std

    for i in range(len(val_X)):
        for j in range(8):
            val_X[i, j, 0] = (val_X[i, j, 0] - u) / std

    for i in range(len(test_X)):
        for j in range(8):
            test_X[i, j, 0] = (test_X[i, j, 0] - u) / std

    u = np.mean(train_val_y[:,:], dtype=np.float32)
    std = np.std(train_val_y[:,:], dtype=np.float32)
    np.save("os_u.npy", u)
    np.save("os_std.npy", std)
    for i in range(len(train_y)):
        for j in range(8):
            train_y[i, j] = (train_y[i, j] - u) / std
    for i in range(len(val_y)):
        for j in range(8):
            val_y[i, j] = (val_y[i, j] - u) / std
    for i in range(len(test_y)):
        for j in range(8):
            test_y[i, j] = (test_y[i, j] - u) / std
    shape = train_X.shape

    np.save('train_X.npy', train_X)
    np.save('train_y.npy', train_y)
    np.save('val_X.npy', val_X)
    np.save('val_y.npy', val_y)
    np.save('test_X.npy', test_X)
    np.save('test_y.npy', test_y)
    # train model
    units = 12
    drop = 0.3
    batch_size = 1
    model = trainwithval([train_X, train_y], [val_X, val_y], units=units[i],
                         drop_out=drop, batch_size=1, epochs=150, L1_value=L1_v, L1_name=L1_array)
    print(datetime.datetime.now())
