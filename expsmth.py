from statsmodels.tsa.holtwinters import SimpleExpSmoothing
import numpy as np
from matplotlib import pyplot
from pandas.plotting import autocorrelation_plot
from pandas import DataFrame
from sklearn.metrics import mean_squared_error, r2_score
from math import sqrt

if __name__ == "__main__":


    # load dataset
    name = "bio"
    aru = 3
    data = np.load(r"cbma\%s_aru%02d.npy" % (name, aru), allow_pickle=True)
    values = data[:, 1]
    # for i in range(len(values)//8):
    #     print(data[i*8+7,0])
    scaled = values[:, np.newaxis].astype('float32')

    train_len = round(len(values)//8 * 0.8)
    trainset = values[:train_len*8]
    testset = values[train_len*8:]
    model = SimpleExpSmoothing(trainset.astype(np.float32), initialization_method='estimated')
    model_fit = model.fit()
    print(model_fit.summary())
    residuals = DataFrame(model_fit.resid)
    residuals.plot()
    # pyplot.show()
    residuals.plot(kind='kde')
    # pyplot.show()
    print(residuals.describe())

    history = [x for x in trainset]
    history = np.array(history)
    predictions = np.array([])
    # walk-forward validation
    for t in range(len(testset)//8):
        model = SimpleExpSmoothing(history.astype(np.float32), initialization_method='estimated')
        model_fit = model.fit()
        output = model_fit.forecast(steps=8)
        yhat = output[:]
        predictions = np.concatenate((predictions,yhat), axis = 0)
        obs = testset[8*t:8*(t+1)]
        history = np.concatenate((history,obs), axis = 0)

    # evaluate forecasts
    rmse = sqrt(mean_squared_error(testset, predictions))
    r2 = r2_score(testset, predictions)
    print('Test RMSE: %.3f' % rmse)
    print('Test R2: %.3f' % r2)
    # plot forecasts against actual outcomes
    pyplot.plot(testset)
    pyplot.plot(predictions, color='red')
    # pyplot.show()