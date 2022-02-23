
from sklearn.metrics import mean_squared_error
import numpy as np
import pmdarima as pm
from math import sqrt

if __name__ == "__main__":
    # load dataset
    name = "bio"
    aru = 4
    data = np.load(r"cbma\%s_aru%02d.npy" % (name, aru), allow_pickle=True)
    values = data[:, 1]
    # for i in range(len(values)//8):
    #     print(data[i*8+7,0])
    scaled = values[:, np.newaxis].astype('float32')
    # autocorrelation_plot(values)
    # pyplot.show()

    train_len = round(len(values)//8 * 0.6)
    val_len = round(len(values)//8 * 0.2)
    test_len = len(values)//8 - train_len - val_len
    trainset = values[:train_len*8]
    valset = values[train_len*8:(train_len+val_len)*8]
    trainval = values[0:(train_len+val_len)*8]
    testset = values[(train_len + val_len)*8:]
    arima = pm.auto_arima(trainval, start_p=2, start_q=2, max_p=5, max_q=5, seasonal= False,
                          out_of_sample_size=val_len, suppress_warnings=True,
                          stepwise=True, error_action='ignore')

    predict_all = np.array([])
    for i in range(test_len):
        each_test = testset[i*8:(i+1)*8]
        preds, conf_int = arima.predict(n_periods=8,
                                        return_conf_int=True)
        arima.update(each_test)
        predict_all = np.concatenate((predict_all,preds), axis = 0)
    print(sqrt(mean_squared_error(testset,predict_all)))



