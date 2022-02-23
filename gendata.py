# prepare data for lstm
from pandas import read_csv
from pandas import DataFrame
from pandas import concat
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
import numpy as np


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    x = np.empty(shape = (0,8,1))
    y = np.empty(shape = (0,8))
    for i in range(len(data) // 8 - 1):
        x = np.concatenate((x, [data[8*i:8*(i+1),:]]),axis = 0)
        y = np.concatenate((y, [data[8*(i+1):8*(i+2), 0]]), axis = 0)
    return x,y

if __name__ == "__main__":
    for name in ["bio"]:
        for aru in range(1,5):
            #load raw data
            dataset = read_csv(r'.\cbma_acoustic_metrics\%s_results_processed_aru%02d.csv'%(name, aru), index_col=0)
            data = dataset[['filename', 'left_channel']]
            raw = data.iloc[:,:]
            start = 0
            tmp = []
            cnt = 0
            for i in range(len(raw)):
                if (raw.iloc[i, 0].split("_")[1].split(".")[0][0:2] == "05"):
                    start = 1
                    cnt = 6
                    tmp.append(raw.iloc[i, 0:2])
                elif start == 1:
                    if (raw.iloc[i, 0].split("_")[1].split(".")[0][0:2] == "%02d" % cnt):
                        if cnt < 12:
                            tmp.append(raw.iloc[i, 0:2])
                            cnt += 1
                        else:
                            tmp.append(raw.iloc[i, 0:2])
                            start = 0
                    else:
                        start = 0
                else:
                    continue
            np.save(r"cbma\%s_aru%02d.npy"%(name, aru), tmp)


