from LSTMstd import create_model
import numpy as np
from pickle import load
import math
from sklearn.metrics import mean_squared_error, r2_score


if __name__ == "__main__":
    L1_array = "1e-6"
    print(L1_array)
    test_X = np.load('test_X.npy')
    test_y = np.load('test_y.npy')
    val_X = np.load('val_X.npy')
    val_y = np.load('val_y.npy')
    train_X = np.load('train_X.npy')
    train_y = np.load('train_y.npy')
    u = np.load("os_u.npy")
    std = np.load("os_std.npy")
    shape = test_X.shape
    batch_size = 1
    val_loss = np.load("std/val_epoch_cmba_bio_aru03_L%s_lstm.npy"%L1_array, allow_pickle=True)
    idx = np.argmin(val_loss[:, 1])
    model = create_model(SLSTM = False, units = 15, drop_out=0.)
    model.load_weights(r'./lstmrecord/train_cmba_bio_aru03_SLSTM_L%s_u15_std_lstm/01_'%L1_array+'%.6f.ckpt'%val_loss[idx,2][0])
    train_len = len(train_y)
    test_len = len(test_y)
    val_len = len(val_y)
    train_y[:,:] = train_y[:, :] * std + u
    val_y[:, :] = val_y[:, :] * std + u
    test_y[:, :] = test_y[:, :] * std + u
    try_X = np.concatenate((train_X,val_X,test_X), axis = 0)
    try_y = np.concatenate((train_y,val_y,test_y), axis=0)
    tryyhat = model.predict(try_X, batch_size = 1)
    tryyhat[:, :] = tryyhat[:, :] * std + u
    rmse = math.sqrt(mean_squared_error(try_y[:train_len, :], tryyhat[:train_len, :]))

    R2 = r2_score(try_y[:train_len, :], tryyhat[:train_len, :])
    print('Train RMSE: %.3f' % rmse)
    print("Train R2 Score: %.3f" % R2)
    rmse = math.sqrt(mean_squared_error(try_y[-test_len:, :], tryyhat[-test_len:, :]))
    R2 = r2_score(try_y[-test_len:, :], tryyhat[-test_len:, :])
    print('Test RMSE: %.3f' % rmse)
    print("Test R2 Score: %.3f" % R2)

    np.save('inv_trainy.npy',try_y[:train_len, :])
    np.save('inv_trainyhat.npy', tryyhat[:train_len, :])
    np.save('inv_valy.npy',try_y[train_len:train_len+val_len, :])
    np.save('inv_valyhat.npy', tryyhat[train_len:train_len+val_len, :])
    np.save('inv_y.npy',try_y[-test_len:, :])
    np.save('inv_yhat.npy', tryyhat[-test_len:, :])

