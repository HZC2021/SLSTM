import numpy as np
import math
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def MSE(pred, label):
    assert len(pred) == len(label), "length not match"
    mse = 0
    for i in range(len(label)):
        mse += (pred[i] - label[i]) * (pred[i] - label[i])
    mse /= len(label)
    return mse

def MAD(pred, label):
    assert len(pred) == len(label), "length not match"
    err = 0
    for i in range(len(label)):
        err += abs(pred[i] - label[i])
    err /= len(label)
    return err

def R2(pred, label):
    return r2_score(label, pred)

def ave_err(pred, label):
    err = 0
    for i in range(len(label)):
        err += pred[i] - label[i]
    err /= len(label)
    return err

def hist_err(pred, label):
    #  matplotlib.axes.Axes.hist() 方法的接口
    err = pred - label
    lb = np.min(err)
    ub = np.max(err)
    plt.hist(x=err, bins=100, color='#0504aa'
                                )
    plt.xlabel('residuals')
    plt.ylabel('count')
    plt.xlim(lb, ub)
    plt.show()

if __name__ == "__main__":
    pass


