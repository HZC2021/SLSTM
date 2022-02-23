# A Seasonal Long Short Term Memory Deep Learning Network for Estimating Time Series of Animal Activity Signals
### Requirements
Python 3.8
Tensorflow 2.6.0
Keras 2.6.0

### 1. The datasets we used
https://drive.google.com/drive/folders/1DivTa2FE6vGE9oiEHlzuPRuj9xNCSRuN?usp=sharing

### 2. Test the S-LSTM
Run trainwithstd.py for training models
Run predictwithstd.py for testing

### Sidenote: Using the GPU
By default, tensorflow is multiple-GPU friendly and it automatically distributes the loads. However, you can also manually lock your computation to one or more of the GPUs. (https://www.tensorflow.org/programmers_guide/using_gpu)
