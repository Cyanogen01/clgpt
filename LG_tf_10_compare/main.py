from __future__ import print_function

import warnings
warnings.filterwarnings('ignore')

import os

import numpy as np

np.random.seed(42)

# Keras
from keras.layers import Input, Dense
from keras.optimizers import Adam, SGD
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.losses import MSE
from keras import losses
# Dataset interfaces
from kgp.datasets.data_utils import data_to_seq

# Model assembling and executing
from kgp.utils.assemble import load_NN_configs, load_GP_configs, assemble
from kgp.utils.experiment import train

# Metrics & losses
from kgp.losses import gen_gp_loss
from kgp.metrics import root_mean_squared_error as RMSE
import csv
import matplotlib.pyplot as plt
import re
import numpy as np


from keras import layers
from kgp.layers import GP
from kgp.models import Model
from keras.initializers import RandomNormal
from keras.layers.advanced_activations import LeakyReLU

import copy
import time

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel

from keras.models import Model as KerasModel

plt.rc('font', family='Times New Roman')
plt.rcParams['axes.linewidth'] = 1.5

def main():
    lag_len = 128
    t_sw_step = 10
    source_Temp = 10
    tf_1_Temp = 25
    tf_2_Temp = 0
    tf_3_Temp = -10
    tf_4_Temp = -20
    train_dir = r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(source_Temp) + r'deg\train_100_sparse'
    test_dir = r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(source_Temp) + r'deg\test_100_sparse'
    tf_train_dir_1 = r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_1_Temp) + r'deg\train_100_sparse'
    tf_test_dir_1 = r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_1_Temp) + r'deg\train_100_sparse_c2'
    tf_train_dir_2 = r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_2_Temp) + r'deg\train_100_sparse'
    tf_test_dir_2 = r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_2_Temp) + r'deg\train_100_sparse_c2'
    tf_train_dir_3 = r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_3_Temp) + r'deg\train_100_sparse'
    tf_test_dir_3 = r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_3_Temp) + r'deg\train_100_sparse_c2'
    tf_train_dir_4 = r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_4_Temp) + r'deg\train_100_sparse_c2'
    tf_test_dir_4 = r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_4_Temp) + r'deg\train_100_sparse_c2'

    """source domain"""
    train_file = os.listdir(train_dir)
    X_seq_train, y_seq_train, X_gp_train, y_gp_train = [], [], [], []
    X_seq_train = np.asarray(X_seq_train).reshape((-1, lag_len, 3, 1))
    y_seq_train = np.asarray(y_seq_train).reshape((-1, 1, 1))
    X_gp_train = np.asarray(X_gp_train).reshape((-1, 3))
    y_gp_train = np.asarray(y_gp_train).reshape((-1, 1))
    for file in train_file:
        index, Current, Voltage, Ah, Temp = [], [], [], [], []
        with open(os.path.join(train_dir, file), 'r') as f:
            reader = csv.reader(f)
            head = next(reader)
            count = 0
            for row in reader:
                Current.append(float(row[1]))
                Voltage.append(float(row[0]))
                Ah.append(float(row[3]))
                Temp.append(float(row[2]))
        index = np.array(index).reshape(-1, 1)
        zero_index = np.max(np.argwhere(Current))
        Current = np.array(Current)[:zero_index].reshape(-1, 1) / 8 + 0.5
        Voltage = np.array(Voltage)[:zero_index].reshape(-1, 1) - 3
        Ah = - np.array(Ah)[:zero_index].reshape(-1, 1)
        Temp = np.array(Temp)[:zero_index].reshape(-1, 1) - source_Temp

        X = np.append(np.append(Current, Voltage, axis=1), Temp, axis=1)
        # y = (Ah - np.min(Ah)) / (np.max(Ah) - np.min(Ah))
        y = Ah / 3.0
        X_seq_train_part, y_seq_train_part = data_to_seq(X, y,
                                                         t_lag=lag_len, t_future_shift=1, t_future_steps=1,
                                                         t_sw_step=t_sw_step)
        X_gp_train_part, y_gp_train_part = X[::t_sw_step], y[::t_sw_step]
        X_seq_train = np.append(X_seq_train, X_seq_train_part.reshape((-1, lag_len, 3, 1)), axis=0)
        y_seq_train = np.append(y_seq_train, y_seq_train_part, axis=0)
        X_gp_train = np.append(X_gp_train, X_gp_train_part, axis=0)
        y_gp_train = np.append(y_gp_train, y_gp_train_part, axis=0)

    index, Current, Voltage, Ah, Temp = [], [], [], [], []
    X_seq_test, y_seq_test, X_gp_test, y_gp_test = [], [], [], []
    X_seq_test = np.asarray(X_seq_test).reshape((-1, lag_len, 3, 1))
    y_seq_test = np.asarray(y_seq_test).reshape((-1, 1, 1))

    with open(r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(
            source_Temp) + r'deg\test_100_sparse\LA92_100_sparse.csv', 'r') as f:
        reader = csv.reader(f)
        head = next(reader)

        for row in reader:
            Current.append(float(row[1]))
            Voltage.append(float(row[0]))
            Ah.append(float(row[3]))
            Temp.append(float(row[2]))
            # if count > 10000:
            # break
            count = count + 1
    index = np.array(index).reshape(-1, 1)
    zero_index = np.max(np.argwhere(Current))
    Current = np.array(Current)[:zero_index].reshape(-1, 1) / 8 + 0.5  # 归一化
    Voltage = np.array(Voltage)[:zero_index].reshape(-1, 1) - 3
    Ah = - np.array(Ah)[:zero_index].reshape(-1, 1)
    Temp = np.array(Temp)[:zero_index].reshape(-1, 1) - source_Temp

    X = np.append(np.append(Current, Voltage, axis=1), Temp, axis=1)
    y = Ah / 3.0
    X_seq_test_part, y_seq_test_part = data_to_seq(X, y,
                                                   t_lag=lag_len, t_future_shift=1, t_future_steps=1,
                                                   t_sw_step=t_sw_step)
    X_seq_test = np.append(X_seq_test, X_seq_test_part.reshape((-1, lag_len, 3, 1)), axis=0)
    y_seq_test = np.append(y_seq_test, y_seq_test_part, axis=0)
    X_gp_test, y_gp_test = X[::t_sw_step], y[::t_sw_step]

    data = {
        'train': [X_seq_train, y_seq_train],
        'valid': [X_seq_test, y_seq_test],
        'test': [X_seq_test, y_seq_test],
    }

    data_LSTM = {
        'train': [X_seq_train.squeeze(), y_seq_train[:, :, 0]],
        'valid': [X_seq_test.squeeze(), y_seq_test[:, :, 0]],
        'test': [X_seq_test.squeeze(), y_seq_test[:, :, 0]],
    }

    """target tf 1"""
    tf_train_file = os.listdir(tf_train_dir_1)
    X_seq_tf_train, y_seq_tf_train, X_gp_tf_train, y_gp_tf_train = [], [], [], []
    X_seq_tf_train = np.asarray(X_seq_tf_train).reshape((-1, lag_len, 3, 1))
    y_seq_tf_train = np.asarray(y_seq_tf_train).reshape((-1, 1, 1))
    X_gp_tf_train = np.asarray(X_gp_tf_train).reshape((-1, 3))
    y_gp_tf_train = np.asarray(y_gp_tf_train).reshape((-1, 1))
    for file in tf_train_file:
        index, Current, Voltage, Ah, Temp = [], [], [], [], []
        with open(os.path.join(tf_train_dir_1, file), 'r') as f:
            reader = csv.reader(f)
            head = next(reader)
            count = 0
            for row in reader:
                Current.append(float(row[1]))
                Voltage.append(float(row[0]))
                Ah.append(float(row[3]))
                Temp.append(float(row[2]))
        index = np.array(index).reshape(-1, 1)
        zero_index = np.max(np.argwhere(Current))
        Current = np.array(Current)[:zero_index].reshape(-1, 1) / 8 + 0.5
        Voltage = np.array(Voltage)[:zero_index].reshape(-1, 1) - 3
        Ah = - np.array(Ah)[:zero_index].reshape(-1, 1)
        Temp = np.array(Temp)[:zero_index].reshape(-1, 1) - tf_1_Temp 

        X = np.append(np.append(Current, Voltage, axis=1), Temp, axis=1)
        y = Ah / 3.0
        X_seq_tf_train_part, y_seq_tf_train_part = data_to_seq(X, y,
                                                               t_lag=lag_len, t_future_shift=1, t_future_steps=1,
                                                               t_sw_step=t_sw_step)
        X_gp_tf_train_part, y_gp_tf_train_part = X[::t_sw_step], y[::t_sw_step]
        X_seq_tf_train = np.append(X_seq_tf_train, X_seq_tf_train_part.reshape((-1, lag_len, 3, 1)), axis=0)
        y_seq_tf_train = np.append(y_seq_tf_train, y_seq_tf_train_part, axis=0)
        X_gp_tf_train = np.append(X_gp_tf_train, X_gp_tf_train_part, axis=0)
        y_gp_tf_train = np.append(y_gp_tf_train, y_gp_tf_train_part, axis=0)

    index, Current, Voltage, Ah, Temp = [], [], [], [], []
    X_seq_tf_test, y_seq_tf_test, X_gp_tf_test, y_gp_tf_test = [], [], [], []
    X_seq_tf_test = np.asarray(X_seq_tf_test).reshape((-1, lag_len, 3, 1))
    y_seq_tf_test = np.asarray(y_seq_tf_test).reshape((-1, 1, 1))
    with open(
            r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_1_Temp) + r'deg\test_100_sparse\LA92_100_sparse.csv',
            'r') as f:
        reader = csv.reader(f)
        head = next(reader)

        for row in reader:
            Current.append(float(row[1]))
            Voltage.append(float(row[0]))
            Ah.append(float(row[3]))
            Temp.append(float(row[2]))
            # if count > 10000:
            # break
            count = count + 1
    index = np.array(index).reshape(-1, 1)
    zero_index = np.max(np.argwhere(Current))
    Current = np.array(Current)[:zero_index].reshape(-1, 1) / 8 + 0.5  # 归一化
    Voltage = np.array(Voltage)[:zero_index].reshape(-1, 1) - 3
    Ah = - np.array(Ah)[:zero_index].reshape(-1, 1)
    Temp = np.array(Temp)[:zero_index].reshape(-1, 1) - tf_1_Temp

    print('Current',len(Current))
    X = np.append(np.concatenate((Current, Voltage), axis=1), Temp, axis=1)
    y = Ah / 3.0
    X_seq_tf_test_part, y_seq_tf_test_part = data_to_seq(X, y,
                                                         t_lag=lag_len, t_future_shift=1, t_future_steps=1,
                                                         t_sw_step=t_sw_step)
    X_seq_tf_test = np.append(X_seq_tf_test, X_seq_tf_test_part.reshape((-1, lag_len, 3, 1)), axis=0)
    y_seq_tf_test = np.append(y_seq_tf_test, y_seq_tf_test_part, axis=0)
    X_gp_tf_test, y_gp_tf_test = X[::t_sw_step], y[::t_sw_step]


    data_tf_1 = {
        'train': [X_seq_tf_train, y_seq_tf_train],
        'valid': [X_seq_tf_test, y_seq_tf_test],
        'test': [X_seq_tf_test, y_seq_tf_test],
    }

    data_LSTM_tf_1 = {
        'train': [X_seq_tf_train.squeeze(), y_seq_tf_train[:, :, 0]],
        'valid': [X_seq_tf_test.squeeze(), y_seq_tf_test[:, :, 0]],
        'test': [X_seq_tf_test.squeeze(), y_seq_tf_test[:, :, 0]],
    }

    """target target 2"""
    tf_train_file = os.listdir(tf_train_dir_2)
    X_seq_tf_train, y_seq_tf_train, X_gp_tf_train, y_gp_tf_train = [], [], [], []
    X_seq_tf_train = np.asarray(X_seq_tf_train).reshape((-1, lag_len, 3, 1))
    y_seq_tf_train = np.asarray(y_seq_tf_train).reshape((-1, 1, 1))
    X_gp_tf_train = np.asarray(X_gp_tf_train).reshape((-1, 3))
    y_gp_tf_train = np.asarray(y_gp_tf_train).reshape((-1, 1))
    for file in tf_train_file:
        index, Current, Voltage, Ah, Temp = [], [], [], [], []
        with open(os.path.join(tf_train_dir_2, file), 'r') as f:
            reader = csv.reader(f)
            head = next(reader)
            count = 0
            for row in reader:
                Current.append(float(row[1]))
                Voltage.append(float(row[0]))
                Ah.append(float(row[3]))
                Temp.append(float(row[2]))
        index = np.array(index).reshape(-1, 1)
        zero_index = np.max(np.argwhere(Current))
        Current = np.array(Current)[:zero_index].reshape(-1, 1) / 8 + 0.5
        Voltage = np.array(Voltage)[:zero_index].reshape(-1, 1) - 3
        Ah = - np.array(Ah)[:zero_index].reshape(-1, 1)
        Temp = np.array(Temp)[:zero_index].reshape(-1, 1) - tf_2_Temp

        X = np.append(np.append(Current, Voltage, axis=1), Temp, axis=1)
        y = Ah / 3.0
        X_seq_tf_train_part, y_seq_tf_train_part = data_to_seq(X, y,
                                                         t_lag=lag_len, t_future_shift=1, t_future_steps=1,
                                                         t_sw_step=t_sw_step)
        X_gp_tf_train_part, y_gp_tf_train_part = X[::t_sw_step], y[::t_sw_step]
        X_seq_tf_train = np.append(X_seq_tf_train, X_seq_tf_train_part.reshape((-1, lag_len, 3, 1)), axis=0)
        y_seq_tf_train = np.append(y_seq_tf_train, y_seq_tf_train_part, axis=0)
        X_gp_tf_train = np.append(X_gp_tf_train, X_gp_tf_train_part, axis=0)
        y_gp_tf_train = np.append(y_gp_tf_train, y_gp_tf_train_part, axis=0)


    index, Current, Voltage, Ah, Temp = [], [], [], [], []
    X_seq_tf_test, y_seq_tf_test, X_gp_tf_test, y_gp_tf_test = [], [], [], []
    X_seq_tf_test = np.asarray(X_seq_tf_test).reshape((-1, lag_len, 3, 1))
    y_seq_tf_test = np.asarray(y_seq_tf_test).reshape((-1, 1, 1))
    with open(r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\'+str(tf_2_Temp)+r'deg\test_100_sparse\LA92_100_sparse.csv', 'r') as f:
        reader = csv.reader(f)
        head = next(reader)

        for row in reader:
            Current.append(float(row[1]))
            Voltage.append(float(row[0]))
            Ah.append(float(row[3]))
            Temp.append(float(row[2]))
            # if count > 10000:
            # break
            count = count + 1
    index = np.array(index).reshape(-1, 1)
    zero_index = np.max(np.argwhere(Current))
    Current = np.array(Current)[:zero_index].reshape(-1, 1) / 8 + 0.5  # 归一化
    Voltage = np.array(Voltage)[:zero_index].reshape(-1, 1) - 3
    Ah = - np.array(Ah)[:zero_index].reshape(-1, 1)
    Temp = np.array(Temp)[:zero_index].reshape(-1, 1) - tf_2_Temp

    X = np.concatenate((Current, Voltage,Temp), axis=1)
    y = Ah / 3.0
    X_seq_tf_test_part, y_seq_tf_test_part = data_to_seq(X, y,
                                                   t_lag=lag_len, t_future_shift=1, t_future_steps=1,
                                                   t_sw_step=t_sw_step)
    X_seq_tf_test = np.append(X_seq_tf_test, X_seq_tf_test_part.reshape((-1, lag_len, 3, 1)), axis=0)
    y_seq_tf_test = np.append(y_seq_tf_test, y_seq_tf_test_part, axis=0)
    X_gp_tf_test, y_gp_tf_test = X[::t_sw_step], y[::t_sw_step]

    data_tf_2 = {
        'train': [X_seq_tf_train, y_seq_tf_train],
        'valid': [X_seq_tf_test, y_seq_tf_test],
        'test': [X_seq_tf_test, y_seq_tf_test],
    }

    data_LSTM_tf_2 = {
        'train': [X_seq_tf_train.squeeze(), y_seq_tf_train[:,:,0]],
        'valid': [X_seq_tf_test.squeeze(), y_seq_tf_test[:,:,0]],
        'test': [X_seq_tf_test.squeeze(), y_seq_tf_test[:,:,0]],
    }

    """target target 3"""
    tf_train_file = os.listdir(tf_train_dir_3)
    X_seq_tf_train, y_seq_tf_train, X_gp_tf_train, y_gp_tf_train = [], [], [], []
    X_seq_tf_train = np.asarray(X_seq_tf_train).reshape((-1, lag_len, 3, 1))
    y_seq_tf_train = np.asarray(y_seq_tf_train).reshape((-1, 1, 1))
    X_gp_tf_train = np.asarray(X_gp_tf_train).reshape((-1, 3))
    y_gp_tf_train = np.asarray(y_gp_tf_train).reshape((-1, 1))
    for file in tf_train_file:
        index, Current, Voltage, Ah, Temp = [], [], [], [], []
        with open(os.path.join(tf_train_dir_3, file), 'r') as f:
            reader = csv.reader(f)
            head = next(reader)
            count = 0
            for row in reader:
                Current.append(float(row[1]))
                Voltage.append(float(row[0]))
                Ah.append(float(row[3]))
                Temp.append(float(row[2]))
        index = np.array(index).reshape(-1, 1)
        zero_index = np.max(np.argwhere(Current))
        Current = np.array(Current)[:zero_index].reshape(-1, 1) / 8 + 0.5
        Voltage = np.array(Voltage)[:zero_index].reshape(-1, 1) - 3
        Ah = - np.array(Ah)[:zero_index].reshape(-1, 1)
        Temp = np.array(Temp)[:zero_index].reshape(-1, 1) - tf_3_Temp

        X = np.append(np.append(Current, Voltage, axis=1), Temp, axis=1)
        y = Ah / 3.0
        X_seq_tf_train_part, y_seq_tf_train_part = data_to_seq(X, y,
                                                               t_lag=lag_len, t_future_shift=1, t_future_steps=1,
                                                               t_sw_step=t_sw_step)
        X_gp_tf_train_part, y_gp_tf_train_part = X[::t_sw_step], y[::t_sw_step]
        X_seq_tf_train = np.append(X_seq_tf_train, X_seq_tf_train_part.reshape((-1, lag_len, 3, 1)), axis=0)
        y_seq_tf_train = np.append(y_seq_tf_train, y_seq_tf_train_part, axis=0)
        X_gp_tf_train = np.append(X_gp_tf_train, X_gp_tf_train_part, axis=0)
        y_gp_tf_train = np.append(y_gp_tf_train, y_gp_tf_train_part, axis=0)

    index, Current, Voltage, Ah, Temp = [], [], [], [], []
    X_seq_tf_test, y_seq_tf_test, X_gp_tf_test, y_gp_tf_test = [], [], [], []
    X_seq_tf_test = np.asarray(X_seq_tf_test).reshape((-1, lag_len, 3, 1))
    y_seq_tf_test = np.asarray(y_seq_tf_test).reshape((-1, 1, 1))
    with open(
            r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_3_Temp) + r'deg\test_100_sparse\LA92_100_sparse.csv',
            'r') as f:
        reader = csv.reader(f)
        head = next(reader)

        for row in reader:
            Current.append(float(row[1]))
            Voltage.append(float(row[0]))
            Ah.append(float(row[3]))
            Temp.append(float(row[2]))
            # if count > 10000:
            # break
            count = count + 1
    index = np.array(index).reshape(-1, 1)
    zero_index = np.max(np.argwhere(Current))
    Current = np.array(Current)[:zero_index].reshape(-1, 1) / 8 + 0.5  # 归一化
    Voltage = np.array(Voltage)[:zero_index].reshape(-1, 1) - 3
    Ah = - np.array(Ah)[:zero_index].reshape(-1, 1)
    Temp = np.array(Temp)[:zero_index].reshape(-1, 1) - tf_3_Temp

    X = np.concatenate((Current, Voltage, Temp), axis=1)
    y = Ah / 3.0
    X_seq_tf_test_part, y_seq_tf_test_part = data_to_seq(X, y,
                                                         t_lag=lag_len, t_future_shift=1, t_future_steps=1,
                                                         t_sw_step=t_sw_step)
    X_seq_tf_test = np.append(X_seq_tf_test, X_seq_tf_test_part.reshape((-1, lag_len, 3, 1)), axis=0)
    y_seq_tf_test = np.append(y_seq_tf_test, y_seq_tf_test_part, axis=0)
    X_gp_tf_test, y_gp_tf_test = X[::t_sw_step], y[::t_sw_step]

    data_tf_3 = {
        'train': [X_seq_tf_train, y_seq_tf_train],
        'valid': [X_seq_tf_test, y_seq_tf_test],
        'test': [X_seq_tf_test, y_seq_tf_test],
    }

    data_LSTM_tf_3 = {
        'train': [X_seq_tf_train.squeeze(), y_seq_tf_train[:,:,0]],
        'valid': [X_seq_tf_test.squeeze(), y_seq_tf_test[:,:,0]],
        'test': [X_seq_tf_test.squeeze(), y_seq_tf_test[:,:,0]],
    }


    """target target 4"""
    tf_train_file = os.listdir(tf_train_dir_4)
    X_seq_tf_train, y_seq_tf_train, X_gp_tf_train, y_gp_tf_train = [], [], [], []
    X_seq_tf_train = np.asarray(X_seq_tf_train).reshape((-1, lag_len, 3, 1))
    y_seq_tf_train = np.asarray(y_seq_tf_train).reshape((-1, 1, 1))
    X_gp_tf_train = np.asarray(X_gp_tf_train).reshape((-1, 3))
    y_gp_tf_train = np.asarray(y_gp_tf_train).reshape((-1, 1))
    for file in tf_train_file:
        index, Current, Voltage, Ah, Temp = [], [], [], [], []
        with open(os.path.join(tf_train_dir_4, file), 'r') as f:
            reader = csv.reader(f)
            head = next(reader)
            count = 0
            for row in reader:
                Current.append(float(row[1]))
                Voltage.append(float(row[0]))
                Ah.append(float(row[3]))
                Temp.append(float(row[2]))
        index = np.array(index).reshape(-1, 1)
        zero_index = np.max(np.argwhere(Current))
        Current = np.array(Current)[:zero_index].reshape(-1, 1) / 8 + 0.5
        Voltage = np.array(Voltage)[:zero_index].reshape(-1, 1) - 3
        Ah = - np.array(Ah)[:zero_index].reshape(-1, 1)
        Temp = np.array(Temp)[:zero_index].reshape(-1, 1) - tf_4_Temp

        X = np.append(np.append(Current, Voltage, axis=1), Temp, axis=1)
        y = Ah / 3.0
        X_seq_tf_train_part, y_seq_tf_train_part = data_to_seq(X, y,
                                                               t_lag=lag_len, t_future_shift=1, t_future_steps=1,
                                                               t_sw_step=t_sw_step)
        X_gp_tf_train_part, y_gp_tf_train_part = X[::t_sw_step], y[::t_sw_step]
        X_seq_tf_train = np.append(X_seq_tf_train, X_seq_tf_train_part.reshape((-1, lag_len, 3, 1)), axis=0)
        y_seq_tf_train = np.append(y_seq_tf_train, y_seq_tf_train_part, axis=0)
        X_gp_tf_train = np.append(X_gp_tf_train, X_gp_tf_train_part, axis=0)
        y_gp_tf_train = np.append(y_gp_tf_train, y_gp_tf_train_part, axis=0)

    index, Current, Voltage, Ah, Temp = [], [], [], [], []
    X_seq_tf_test, y_seq_tf_test, X_gp_tf_test, y_gp_tf_test = [], [], [], []
    X_seq_tf_test = np.asarray(X_seq_tf_test).reshape((-1, lag_len, 3, 1))
    y_seq_tf_test = np.asarray(y_seq_tf_test).reshape((-1, 1, 1))
    with open(
            r'C:\Users\iCosMea Pro\SOC estimation\csv_data_tte\LG\\' + str(tf_4_Temp) + r'deg\test_100_sparse\LA92_100_sparse.csv',
            'r') as f:
        reader = csv.reader(f)
        head = next(reader)

        for row in reader:
            Current.append(float(row[1]))
            Voltage.append(float(row[0]))
            Ah.append(float(row[3]))
            Temp.append(float(row[2]))
            # if count > 10000:
            # break
            count = count + 1
    index = np.array(index).reshape(-1, 1)
    zero_index = np.max(np.argwhere(Current))
    Current = np.array(Current)[:zero_index].reshape(-1, 1) / 8 + 0.5  # 归一化
    Voltage = np.array(Voltage)[:zero_index].reshape(-1, 1) - 3
    Ah = - np.array(Ah)[:zero_index].reshape(-1, 1)
    Temp = np.array(Temp)[:zero_index].reshape(-1, 1) - tf_4_Temp

    X = np.concatenate((Current, Voltage, Temp), axis=1)
    y = Ah / 3.0
    X_seq_tf_test_part, y_seq_tf_test_part = data_to_seq(X, y,
                                                         t_lag=lag_len, t_future_shift=1, t_future_steps=1,
                                                         t_sw_step=t_sw_step)
    X_seq_tf_test = np.append(X_seq_tf_test, X_seq_tf_test_part.reshape((-1, lag_len, 3, 1)), axis=0)
    y_seq_tf_test = np.append(y_seq_tf_test, y_seq_tf_test_part, axis=0)
    X_gp_tf_test, y_gp_tf_test = X[::t_sw_step], y[::t_sw_step]

    data_tf_4 = {
        'train': [X_seq_tf_train, y_seq_tf_train],
        'valid': [X_seq_tf_test, y_seq_tf_test],
        'test': [X_seq_tf_test, y_seq_tf_test],
    }
    data_LSTM_tf_4 = {
        'train': [X_seq_tf_train.squeeze(), y_seq_tf_train[:,:,0]],
        'valid': [X_seq_tf_test.squeeze(), y_seq_tf_test[:,:,0]],
        'test': [X_seq_tf_test.squeeze(), y_seq_tf_test[:,:,0]],
    }


    """reformat"""
    for set_name in data:
        y = data[set_name][1]
        y = y.reshape((-1, 1, np.prod(y.shape[1:])))
        data[set_name][1] = [y[:, :, i] for i in range(y.shape[2])]

    for set_name in data_tf_1:
        y = data_tf_1[set_name][1]
        y = y.reshape((-1, 1, np.prod(y.shape[1:])))
        data_tf_1[set_name][1] = [y[:, :, i] for i in range(y.shape[2])]

    for set_name in data_tf_2:
        y = data_tf_2[set_name][1]
        y = y.reshape((-1, 1, np.prod(y.shape[1:])))
        data_tf_2[set_name][1] = [y[:, :, i] for i in range(y.shape[2])]

    for set_name in data_tf_3:
        y = data_tf_3[set_name][1]
        y = y.reshape((-1, 1, np.prod(y.shape[1:])))
        data_tf_3[set_name][1] = [y[:, :, i] for i in range(y.shape[2])]

    for set_name in data_tf_4:
        y = data_tf_4[set_name][1]
        y = y.reshape((-1, 1, np.prod(y.shape[1:])))
        data_tf_4[set_name][1] = [y[:, :, i] for i in range(y.shape[2])]

    X_train = np.asarray(data['train'][0])
    X_test = np.asarray(data['test'][0])
    X_valid = np.asarray(data['valid'][0])
    y_train = np.asarray(data['train'][1]).flatten()
    y_test = np.asarray(data['test'][1]).flatten()
    y_valid = np.asarray(data['valid'][1]).flatten()

    X_train_tf_1 = np.asarray(data_tf_1['train'][0])
    X_test_tf_1 = np.asarray(data_tf_1['test'][0])
    X_valid_tf_1 = np.asarray(data_tf_1['valid'][0])
    y_train_tf_1 = np.asarray(data_tf_1['train'][1]).flatten()
    y_test_tf_1 = np.asarray(data_tf_1['test'][1]).flatten()
    y_valid_tf_1 = np.asarray(data_tf_1['valid'][1]).flatten()

    X_train_tf_2 = np.asarray(data_tf_2['train'][0])
    X_test_tf_2 = np.asarray(data_tf_2['test'][0])
    X_valid_tf_2 = np.asarray(data_tf_2['valid'][0])
    y_train_tf_2 = np.asarray(data_tf_2['train'][1]).flatten()
    y_test_tf_2 = np.asarray(data_tf_2['test'][1]).flatten()
    y_valid_tf_2 = np.asarray(data_tf_2['valid'][1]).flatten()

    X_train_tf_3 = np.asarray(data_tf_3['train'][0])
    X_test_tf_3 = np.asarray(data_tf_3['test'][0])
    X_valid_tf_3 = np.asarray(data_tf_3['valid'][0])
    y_train_tf_3 = np.asarray(data_tf_3['train'][1]).flatten()
    y_test_tf_3 = np.asarray(data_tf_3['test'][1]).flatten()
    y_valid_tf_3 = np.asarray(data_tf_3['valid'][1]).flatten()

    X_train_tf_4 = np.asarray(data_tf_4['train'][0])
    X_test_tf_4 = np.asarray(data_tf_4['test'][0])
    X_valid_tf_4 = np.asarray(data_tf_4['valid'][0])
    y_train_tf_4 = np.asarray(data_tf_4['train'][1]).flatten()
    y_test_tf_4 = np.asarray(data_tf_4['test'][1]).flatten()
    y_valid_tf_4 = np.asarray(data_tf_4['valid'][1]).flatten()

    """Network construction"""
    nb_train_samples = data['train'][0].shape[0]
    input_shape = list(data['train'][0].shape[1:])
    output_shape = list(data['train'][1][0].shape[1:])
    nb_outputs = len(data['train'][1])
    gp_input_shape = (1,)
    batch_size = 128
    epochs = 1  # epoch:1000 lr:-3 len:64 dim:64 -> 0.008/0.25
    # 0:1 1000 -4 128 128 0.0067 0.251
    LSMT_epochs = 1
    CNN_LSTM_epoch = 1
    CNN_epoch = 1
    GRU_epochs = 1
    CNN_GRU_epoch = 1000

    LSMT_epochs_tf = 100
    tf_write_flag = False

    nn_params = {
        'H_dim': 64,  # 经测试，Hdim不影响LSTM输出个数, LSTM输出个数为总样本周期数 64
        'H_activation': 'tanh',
        'dropout': 0.1,
    }

    gp_params = {
        'cov': 'SEiso',
        # 'hyp_lik': -2.0,
        'hyp_lik': -0.0,
        # 'hyp_mean': [[1.0], [1.0]],
        'hyp_cov': [[-0.7], [0.0]],
        # 'hyp_cov': [[0.0], [0.0]],
        # 'hyp_cov': [[-0.4], [-0.1]],
        'opt': {'cg_maxit': 20000, 'cg_tol': 1e-4},
        'grid_kwargs': {'eq': 1, 'k': 1e2},
        'update_grid': True,
    }

    nn_configs = load_NN_configs(filename='gru.yaml',
                                 input_shape=input_shape,
                                 output_shape=gp_input_shape,
                                 params=nn_params)

    nn_configs_lstm = load_NN_configs(filename='lstm.yaml',
                                 input_shape=input_shape,
                                 output_shape=gp_input_shape,
                                 params=nn_params)

    gp_configs = load_GP_configs(filename='gp.yaml',
                                 nb_outputs=nb_outputs,
                                 batch_size=batch_size,
                                 nb_train_samples=nb_train_samples,
                                 params=gp_params)

    # Construct & compile the model
    
    inputs = layers.Input(shape=input_shape)

    previous = layers.Conv2D(filters=16, kernel_size=(4, 1))(inputs)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=32, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(16,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)

    previous = layers.Conv2D(filters=64, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=64, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(8,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)

    previous = layers.Conv2D(filters=128, kernel_size=(16, 2),  )(previous) # (8,2) -> (8,1) 
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=64, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(16,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)

    previous = layers.Conv2D(filters=64, kernel_size=(4, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=64, kernel_size=(4, 1),   )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(16,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(1,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)


    previous = layers.Reshape([previous.shape[1].value, previous.shape[2].value])(previous)
    previous = layers.GRU(**nn_configs['1H']['hidden_layers'][0]['config'])(previous)
    previous = layers.Dropout(0.1)(previous)
    # previous = layers.BatchNormalization()(previous)
    output_dim = np.prod(output_shape)
    previous = layers.Dense(output_dim)(previous)
    # outputs = [GP(**gp_configs['MSGP']['config'])(previous)]
    outputs = [GP(**gp_configs['MSGP']['config'])(previous)]

    model = Model(inputs=inputs, outputs=outputs)
    # model = assemble('GP-LSTM', [nn_configs['1H'], gp_configs['MSGP']])
    loss = [gen_gp_loss(gp) for gp in model.output_gp_layers]
    model.compile(optimizer=Adam(1e-4), loss=loss)

    model.summary()

    # Callbacks
    save_best = ModelCheckpoint(filepath='weight_best_many_cycle.h5', monitor='val_mse', verbose=1, save_best_only=True,
                                mode='min')
    #callbacks = [EarlyStopping(monitor='nlml', patience=200), save_best]
    callbacks = [EarlyStopping(monitor='nlml', patience=200)]

    train_time_start = time.time()
    history = train(model, data, callbacks=callbacks, gp_n_iter=5,  # gp_n_iter 过大会影响估计精度
                    checkpoint=None, checkpoint_monitor='nlml',
                    epochs=epochs, batch_size=batch_size, verbose=1)
    train_time = time.time() - train_time_start

    model.load_weights('weight_best_many_cycle.h5')


    model_t = KerasModel(inputs=model.input, outputs=model.get_layer('dense_5').output)
    # y = model_t.predict(data['train'][0])[:,:,0,0]
    y = model_t.predict(data['train'][0])[:,0,:,0]
    #y = model_t.predict(data['train'][0])[:,5,0,:]
    #y = model_t.predict(data['train'][0])
    plt.plot(range(len(y)), y)
    plt.plot(range(len(y_train)),y_train.flatten(),c='k')
    # plt.show()
    # assert False

    gp_layer = model.output_gp_layers
    raw_hyp = gp_layer[0].hyp

    print('hyp set: ','\nhyp_cov:',gp_params['hyp_cov'],'\nhyp_lik:',gp_params['hyp_lik'])
    print('\nhyp learnt from source domain: ',raw_hyp)
    
    # X_tr_source = gp_layer[0].backend.eng.pull('X_tr')
    # y_tr_source = gp_layer[0].backend.eng.pull('y_tr')
    
    plt.figure()
    plt.plot(range(len(history.history['loss'])), history.history['loss'], label='loss')
    plt.plot(range(len(history.history['gp_1_mse'])), history.history['gp_1_mse'], label='gp_1_mse')
    plt.plot(range(len(history.history['gp_1_nlml'])), history.history['gp_1_nlml'], label='gp_1_nlml')
    plt.plot(range(len(history.history['mse'])), history.history['mse'], label='mse')
    plt.plot(range(len(history.history['nlml'])), history.history['nlml'], label='nlml')
    plt.plot(range(len(history.history['val_mse'])), history.history['val_mse'], label='val_mse')
    plt.plot(range(len(history.history['val_nlml'])), history.history['val_nlml'], label='val_nlml')
    plt.title('train loss process')
    plt.legend()

    plt.figure()
    H_train_before = model.transform(np.asarray(data['train'][0]), batch_size=batch_size)[0].flatten()
    plt.plot(range(len(H_train_before)), H_train_before)

    fine_time_start = time.time()
    model.finetune(*data['train'],
                   batch_size=batch_size,
                   gp_n_iter=10000,  # gp_n_iter 过大会影响CI
                   verbose=1)
    fine_time = time.time() - fine_time_start
    "source domain hyp"
    fine_hyp = gp_layer[0].hyp

    y_pre_train, s2_train = list(model.predict(data['train'][0], return_var=True))  # 返回平均值和方差  models.py是kgp的包里面的
    rmse_predict_train = RMSE(y_train, y_pre_train)
    print('GP-LSTM ' +str(source_Temp)+ ' train RMSE:', rmse_predict_train)

    y_pre, s2 = list(model.predict(data['test'][0], return_var=True))  # 返回平均值和方差  models.py是kgp的包里面的
    rmse_predict = RMSE(y_test, y_pre)
    print('GP-LSTM ' +str(source_Temp)+ ' Test RMSE:', rmse_predict)

    y_pre_di_tf1, s2_di_tf1 = list(model.predict(data_tf_1['test'][0], return_var=True))
    rmse_predict_di_1 = RMSE(np.asarray(data_tf_1['test'][1]).flatten(), y_pre_di_tf1)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_1_Temp) + 'direct RMSE:', rmse_predict_di_1)

    y_pre_di_tf2, s2_di_tf2 = list(model.predict(data_tf_2['test'][0], return_var=True))
    rmse_predict_di_2 = RMSE(np.asarray(data_tf_2['test'][1]).flatten(), y_pre_di_tf2)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_2_Temp) + ' direct RMSE:', rmse_predict_di_2)

    y_pre_di_tf3, s2_di_tf3 = list(model.predict(data_tf_3['test'][0], return_var=True))
    rmse_predict_di_3 = RMSE(np.asarray(data_tf_3['test'][1]).flatten(), y_pre_di_tf3)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_3_Temp) + ' direct RMSE:', rmse_predict_di_3)

    y_pre_di_tf4, s2_di_tf4 = list(model.predict(data_tf_4['test'][0], return_var=True))
    rmse_predict_di_4 = RMSE(np.asarray(data_tf_4['test'][1]).flatten(), y_pre_di_tf4)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_4_Temp) + ' direct RMSE:', rmse_predict_di_4)

    """transfer to tf1"""
    tf_1_time_start = time.time()
    model.finetune(*data_tf_1['train'],
                   batch_size=batch_size,
                   gp_n_iter=10000,  # gp_n_iter 过大会影响CI
                   verbose=1)
    tf_1_time = time.time() - tf_1_time_start
    hyp_tf_1 = copy.deepcopy(model.output_gp_layers[0].hyp)

    y_pre_tf_1, s2_tf_1 = list(model.predict(data_tf_1['test'][0], return_var=True))
    rmse_predict_tf_1 = RMSE(y_test_tf_1, y_pre_tf_1)
    print('GP-LSTM ' + str(source_Temp) + 'to' + str(tf_1_Temp) + ' Transfer_Test RMSE:', rmse_predict_tf_1)

    print('\nhyp tf1: ',hyp_tf_1)
    X_tr_tf_1 = gp_layer[0].backend.eng.pull('X_tr')
    y_tr_tf_1 = gp_layer[0].backend.eng.pull('y_tr')

    """transfer to tf2"""
    tf_2_time_start = time.time()
    model.finetune(*data_tf_2['train'],
                   batch_size=batch_size,
                   gp_n_iter=10000,  # gp_n_iter 过大会影响CI
                   verbose=1)
    tf_2_time = time.time() - tf_2_time_start
    hyp_tf_2 = copy.deepcopy(model.output_gp_layers[0].hyp)

    y_pre_tf_2, s2_tf_2 = list(model.predict(data_tf_2['test'][0], return_var=True))
    rmse_predict_tf_2 = RMSE(y_test_tf_2, y_pre_tf_2)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_2_Temp) + ' Transfer_Test RMSE:', rmse_predict_tf_2)

    print('\nhyp tf2: ',hyp_tf_2)
    X_tr_tf_2 = gp_layer[0].backend.eng.pull('X_tr')
    y_tr_tf_2 = gp_layer[0].backend.eng.pull('y_tr')

    """transfer to tf3"""
    tf_3_time_start = time.time()
    model.finetune(*data_tf_3['train'],
                   batch_size=batch_size,
                   gp_n_iter=10000,  # gp_n_iter 过大会影响CI
                   verbose=1)
    tf_3_time = time.time() - tf_3_time_start
    hyp_tf_3 = copy.deepcopy(model.output_gp_layers[0].hyp)

    y_pre_tf_3, s2_tf_3 = list(model.predict(data_tf_3['test'][0], return_var=True))
    rmse_predict_tf_3 = RMSE(y_test_tf_3, y_pre_tf_3)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_3_Temp) + ' Transfer_Test RMSE:', rmse_predict_tf_3)

    print('\nhyp tf3: \n',hyp_tf_3)
    X_tr_tf_3 = gp_layer[0].backend.eng.pull('X_tr')
    y_tr_tf_3 = gp_layer[0].backend.eng.pull('y_tr')


    """transfer to tf4"""
    tf_4_time_start = time.time()
    model.finetune(*data_tf_4['train'],
                   batch_size=batch_size,
                   gp_n_iter=10000,  # gp_n_iter 过大会影响CI
                   verbose=1)
    tf_4_time = time.time() - tf_4_time_start
    hyp_tf_4 = copy.deepcopy(model.output_gp_layers[0].hyp)

    y_pre_tf_4, s2_tf_4 = list(model.predict(data_tf_4['test'][0], return_var=True))
    rmse_predict_tf_4 = RMSE(y_test_tf_4, y_pre_tf_4)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_4_Temp) + ' Transfer_Test RMSE:', rmse_predict_tf_4)

    print('\nhyp tf4: ',hyp_tf_4)
    X_tr_tf_4 = gp_layer[0].backend.eng.pull('X_tr')
    y_tr_tf_4 = gp_layer[0].backend.eng.pull('y_tr')

    # with open('GP_GRU_tf.csv', 'w', newline='') as f:
    #     csv_writer = csv.writer(f)
    #     csv_writer.writerow(data_LSTM['test'][1].flatten())
    #     csv_writer.writerow(data_LSTM_tf_1['test'][1].flatten())
    #     csv_writer.writerow(data_LSTM_tf_2['test'][1].flatten())
    #     csv_writer.writerow(data_LSTM_tf_3['test'][1].flatten())
    #     csv_writer.writerow(data_LSTM_tf_4['test'][1].flatten())
    #     csv_writer.writerow(y_pre[0].flatten())
    #     csv_writer.writerow(s2[0].flatten().flatten())
    #     csv_writer.writerow(y_pre_di_tf1[0].flatten())
    #     csv_writer.writerow(s2_di_tf1[0].flatten())
    #     csv_writer.writerow(y_pre_di_tf2[0].flatten())
    #     csv_writer.writerow(s2_di_tf2[0].flatten())
    #     csv_writer.writerow(y_pre_di_tf3[0].flatten())
    #     csv_writer.writerow(s2_di_tf3[0].flatten())
    #     csv_writer.writerow(y_pre_di_tf4[0].flatten())
    #     csv_writer.writerow(s2_di_tf4[0].flatten())
    #     csv_writer.writerow(y_pre_tf_1[0].flatten())
    #     csv_writer.writerow(s2_tf_1[0].flatten())
    #     csv_writer.writerow(y_pre_tf_2[0].flatten())
    #     csv_writer.writerow(s2_tf_2[0].flatten())
    #     csv_writer.writerow(y_pre_tf_3[0].flatten())
    #     csv_writer.writerow(s2_tf_3[0].flatten())
    #     csv_writer.writerow(y_pre_tf_4[0].flatten())
    #     csv_writer.writerow(s2_tf_4[0].flatten()) 
    #     csv_writer.writerow(history.history['nlml'])


    plt.figure()
    plt.plot(range(len(y_train)),y_train.flatten())
    plt.plot(range(len(y_pre_train[0])),y_pre_train[0])


    plt.figure()
    plt.plot(range(len(y_test_tf_1)),y_test_tf_1.flatten())
    plt.plot(range(len(y_pre_tf_1[0])),y_pre_tf_1[0])
    plt.plot(range(len(y_test_tf_1)),y_pre_di_tf1[0])


    plt.figure()
    plt.plot(range(len(y_test_tf_2)),y_test_tf_2.flatten())
    plt.plot(range(len(y_pre_tf_2[0])),y_pre_tf_2[0])
    plt.plot(range(len(y_test_tf_2)),y_pre_di_tf2[0])


    plt.figure()
    plt.plot(range(len(y_test_tf_3)),y_test_tf_3.flatten())
    plt.plot(range(len(y_pre_tf_3[0])),y_pre_tf_3[0])
    plt.plot(range(len(y_test_tf_3)),y_pre_di_tf3[0])


    plt.figure()
    plt.plot(range(len(y_test_tf_4)),y_test_tf_4.flatten())
    plt.plot(range(len(y_pre_tf_4[0])),y_pre_tf_4[0])
    plt.plot(range(len(y_test_tf_4)),y_pre_di_tf4[0])


    """CNN-LSTM"""

    #CNN_LSTM_model = KerasModel(inputs=model.input, outputs=model.get_layer('dense_6').output)
    nb_train_samples = data['train'][0].shape[0]
    input_shape = list(data['train'][0].shape[1:])
    output_shape = list(data['train'][1][0].shape[1:])
    nb_outputs = len(data['train'][1])

    inputs = layers.Input(shape=input_shape)

    previous = layers.Conv2D(filters=16, kernel_size=(4, 1))(inputs)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=32, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(16,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)

    previous = layers.Conv2D(filters=64, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=64, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(8,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)

    previous = layers.Conv2D(filters=128, kernel_size=(16, 2),  )(previous) # (8,2) -> (8,1) 
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=64, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(16,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)

    previous = layers.Conv2D(filters=64, kernel_size=(4, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=64, kernel_size=(4, 1),   )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(16,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(1,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)


    previous = layers.Reshape([previous.shape[1].value, previous.shape[2].value])(previous)
    previous = layers.LSTM(**nn_configs_lstm['1H']['hidden_layers'][0]['config'])(previous)
    previous = layers.Dropout(0.1)(previous)
    # previous = layers.BatchNormalization()(previous)
    output_dim = np.prod(output_shape)
    outputs = layers.Dense(output_dim)(previous)
    CNN_LSTM_model = KerasModel(inputs=inputs, outputs=outputs)

    CNN_LSTM_model.compile(optimizer=Adam(1e-5), loss=MSE)
    print('CNN-LSTM summary')
    CNN_LSTM_model.summary()

    CNN_LSTM_time_start = time.time()
    CNN_LSTM_history = CNN_LSTM_model.fit(data['train'][0], data['train'][1], epochs=CNN_LSTM_epoch)

    CNN_LSTM_time_end = time.time()
    CNN_LSTM_time = CNN_LSTM_time_end - CNN_LSTM_time_start

    y_pred_CNN_LSTM = CNN_LSTM_model.predict(data['test'][0])
    CNN_LSTM_RMSE = RMSE(data['test'][1], y_pred_CNN_LSTM)
    print("CNN LSTM RMSE: ", CNN_LSTM_RMSE)
    plt.figure()
    plt.plot(range(len(data['test'][1][0].flatten())), data['test'][1][0].flatten())
    plt.plot(range(len(y_pred_CNN_LSTM)), y_pred_CNN_LSTM)
    plt.title('CNN-LSTM')

    '''CNN LSTM tf1'''
    CNN_LSTM_model_tf1 = KerasModel(inputs=CNN_LSTM_model.input, outputs=CNN_LSTM_model.output)
    for index, layer in enumerate(CNN_LSTM_model_tf1.layers):
        # print(index, layer)
        if not index > 4:
            layer.trainable = False

    CNN_LSTM_model_tf1.compile(optimizer=Adam(1e-5), loss=MSE)
    CNN_LSTM_time_tf1_start = time.time()
    train(CNN_LSTM_model_tf1, data_tf_1, callbacks=None,
          epochs=LSMT_epochs_tf, batch_size=batch_size, verbose=1)
    CNN_LSTM_time_tf1_end = time.time()
    CNN_LSTM_time_tf1 = CNN_LSTM_time_tf1_end - CNN_LSTM_time_tf1_start
    y_pred_CNN_LSTM_tf1 = CNN_LSTM_model_tf1.predict(data_tf_1['test'][0])

    '''CNN LSTM tf2'''
    CNN_LSTM_model_tf2 = KerasModel(inputs=CNN_LSTM_model.input, outputs=CNN_LSTM_model.output)
    for index, layer in enumerate(CNN_LSTM_model_tf2.layers):
        # print(index, layer)
        if not index > 15:
            layer.trainable = False

    CNN_LSTM_model_tf2.compile(optimizer=Adam(1e-5), loss=MSE)
    CNN_LSTM_time_tf2_start = time.time()
    train(CNN_LSTM_model_tf2, data_tf_2, callbacks=None,
          epochs=LSMT_epochs_tf, batch_size=batch_size, verbose=1)
    CNN_LSTM_time_tf2_end = time.time()
    CNN_LSTM_time_tf2 = CNN_LSTM_time_tf2_end - CNN_LSTM_time_tf2_start
    y_pred_CNN_LSTM_tf2 = CNN_LSTM_model_tf1.predict(data_tf_2['test'][0])

    '''CNN LSTM tf3'''
    CNN_LSTM_model_tf3 = KerasModel(inputs=CNN_LSTM_model.input, outputs=CNN_LSTM_model.output)
    for index, layer in enumerate(CNN_LSTM_model_tf3.layers):
        # print(index, layer)
        if not index > 15:
            layer.trainable = False

    CNN_LSTM_model_tf3.compile(optimizer=Adam(1e-5), loss=MSE)
    CNN_LSTM_time_tf3_start = time.time()
    train(CNN_LSTM_model_tf3, data_tf_3, callbacks=None,
          epochs=LSMT_epochs_tf, batch_size=batch_size, verbose=1)
    CNN_LSTM_time_tf3_end = time.time()
    CNN_LSTM_time_tf3 = CNN_LSTM_time_tf3_end - CNN_LSTM_time_tf3_start
    y_pred_CNN_LSTM_tf3 = CNN_LSTM_model_tf3.predict(data_tf_3['test'][0])

    '''CNN LSTM tf4'''
    CNN_LSTM_model_tf4 = KerasModel(inputs=CNN_LSTM_model.input, outputs=CNN_LSTM_model.output)
    for index, layer in enumerate(CNN_LSTM_model_tf4.layers):
        # print(index, layer)
        if not index > 15:
            layer.trainable = False

    CNN_LSTM_model_tf4.compile(optimizer=Adam(1e-5), loss=MSE)
    CNN_LSTM_time_tf4_start = time.time()
    train(CNN_LSTM_model_tf4, data_tf_4, callbacks=None,
          epochs=LSMT_epochs_tf, batch_size=batch_size, verbose=1)
    CNN_LSTM_time_tf4_end = time.time()
    CNN_LSTM_time_tf4 = CNN_LSTM_time_tf4_end - CNN_LSTM_time_tf4_start
    y_pred_CNN_LSTM_tf4 = CNN_LSTM_model_tf4.predict(data_tf_4['test'][0])

    y_pred_CNN_LSMT_di1 = CNN_LSTM_model.predict(data_tf_1['test'][0])
    y_pred_CNN_LSMT_di2 = CNN_LSTM_model.predict(data_tf_2['test'][0])
    y_pred_CNN_LSMT_di3 = CNN_LSTM_model.predict(data_tf_3['test'][0])
    y_pred_CNN_LSMT_di4 = CNN_LSTM_model.predict(data_tf_4['test'][0])
    if tf_write_flag:
        with open('CNN_LSTM_tf.csv', 'w', newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(y_pred_CNN_LSTM.flatten())
            csv_writer.writerow(y_pred_CNN_LSMT_di1.flatten())
            csv_writer.writerow(y_pred_CNN_LSMT_di2.flatten())
            csv_writer.writerow(y_pred_CNN_LSMT_di3.flatten())
            csv_writer.writerow(y_pred_CNN_LSMT_di4.flatten())
            csv_writer.writerow(y_pred_CNN_LSTM_tf1.flatten())
            csv_writer.writerow(y_pred_CNN_LSTM_tf2.flatten())
            csv_writer.writerow(y_pred_CNN_LSTM_tf3.flatten())
            csv_writer.writerow(y_pred_CNN_LSTM_tf4.flatten()) 
            print(CNN_LSTM_history.history)
            csv_writer.writerow(CNN_LSTM_history.history['loss'])


    """CNN-GRU"""
    nn_params_gru = {
        'H_dim': 64,
        'H_activation': 'tanh',
        'dropout': 0.1,
    }
    #CNN_LSTM_model = KerasModel(inputs=model.input, outputs=model.get_layer('dense_6').output)
    nb_train_samples = data['train'][0].shape[0]
    input_shape = list(data['train'][0].shape[1:])
    output_shape = list(data['train'][1][0].shape[1:])
    nb_outputs = len(data['train'][1])

    nn_configs_gru = load_NN_configs(filename='gru.yaml',
                                 input_shape=input_shape,
                                 output_shape=gp_input_shape,
                                 params=nn_params_gru)

    inputs = layers.Input(shape=input_shape)
    #previous = layers.BatchNormalization()(inputs)

    previous = layers.Conv2D(filters=16, kernel_size=(4, 1))(inputs)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=32, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(16,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)

    previous = layers.Conv2D(filters=64, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=64, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(8,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)

    previous = layers.Conv2D(filters=128, kernel_size=(16, 2),  )(previous) # (8,2) -> (8,1) 
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=64, kernel_size=(8, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(16,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)

    previous = layers.Conv2D(filters=64, kernel_size=(4, 1),  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Conv2D(filters=64, kernel_size=(4, 1),   )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(16,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)
    previous = layers.Dense(1,  )(previous)
    previous = LeakyReLU(alpha=0.3)(previous)


    previous = layers.Reshape([previous.shape[1].value, previous.shape[2].value])(previous)
    previous = layers.GRU(**nn_configs_gru['1H']['hidden_layers'][0]['config'])(previous)
    previous = layers.Dropout(0.1)(previous)
    # previous = layers.BatchNormalization()(previous)
    output_dim = np.prod(output_shape)
    outputs = layers.Dense(output_dim)(previous)
    CNN_GRU_model = KerasModel(inputs=inputs, outputs=outputs)

    CNN_GRU_model.compile(optimizer=Adam(1e-5), loss=MSE)
    print('CNN_GRU_model')
    CNN_GRU_model.summary()

    CNN_GRU_callbacks = [TensorBoard('CNN_GRU')]

    CNN_GRU_time_start = time.time()
    CNN_GRU_history = CNN_GRU_model.fit(data['train'][0],data['train'][1], epochs=CNN_GRU_epoch, callbacks=CNN_GRU_callbacks)
    CNN_GRU_time_end = time.time()
    CNN_GRU_time = CNN_GRU_time_end - CNN_GRU_time_start
    y_pred_CNN_GRU = CNN_GRU_model.predict(data['test'][0])
    CNN_GRU_RMSE = RMSE(data['test'][1], y_pred_CNN_GRU)
    print("CNN GRU RMSE: ", CNN_GRU_RMSE)
    plt.figure()
    plt.plot(range(len(data['test'][1][0].flatten())), data['test'][1][0].flatten())
    plt.plot(range(len(y_pred_CNN_GRU)), y_pred_CNN_GRU)
    plt.title('CNN-GRU')

    '''CNN GRU tf1'''
    CNN_GRU_model_tf1 = KerasModel(inputs=CNN_GRU_model.input, outputs=CNN_GRU_model.output)
    for index, layer in enumerate(CNN_GRU_model_tf1.layers):
        if not index > 15:
            layer.trainable = False

    CNN_GRU_model_tf1.compile(optimizer=Adam(1e-5), loss=MSE)
    CNN_GRU_time_tf1_start = time.time()
    train(CNN_GRU_model_tf1, data_tf_1, callbacks=None,
          epochs=LSMT_epochs_tf, batch_size=batch_size, verbose=1)
    CNN_GRU_time_tf1_end = time.time()
    CNN_GRU_time_tf1 = CNN_GRU_time_tf1_end - CNN_GRU_time_tf1_start
    y_pred_CNN_GRU_tf1 = CNN_GRU_model_tf1.predict(data_tf_1['test'][0])

    '''CNN GRU tf2'''
    CNN_GRU_model_tf2 = KerasModel(inputs=CNN_GRU_model.input, outputs=CNN_GRU_model.output)
    for index, layer in enumerate(CNN_GRU_model_tf2.layers):
        # print(index, layer)
        if not index > 15:
            layer.trainable = False

    CNN_GRU_model_tf2.compile(optimizer=Adam(1e-5), loss=MSE)
    CNN_GRU_time_tf2_start = time.time()
    train(CNN_GRU_model_tf2, data_tf_2, callbacks=None,
          epochs=LSMT_epochs_tf, batch_size=batch_size, verbose=1)
    CNN_GRU_time_tf2_end = time.time()
    CNN_GRU_time_tf2 = CNN_GRU_time_tf2_end - CNN_GRU_time_tf2_start
    y_pred_CNN_GRU_tf2 = CNN_GRU_model_tf2.predict(data_tf_2['test'][0])

    '''CNN GRU tf3'''
    CNN_GRU_model_tf3 = KerasModel(inputs=CNN_GRU_model.input, outputs=CNN_GRU_model.output)
    for index, layer in enumerate(CNN_GRU_model_tf3.layers):
        # print(index, layer)
        if not index > 15:
            layer.trainable = False

    CNN_GRU_model_tf3.compile(optimizer=Adam(1e-5), loss=MSE)
    CNN_GRU_time_tf3_start = time.time()
    train(CNN_GRU_model_tf3, data_tf_3, callbacks=None,
          epochs=LSMT_epochs_tf, batch_size=batch_size, verbose=1)
    CNN_GRU_time_tf3_end = time.time()
    CNN_GRU_time_tf3 = CNN_GRU_time_tf3_end - CNN_GRU_time_tf3_start
    y_pred_CNN_GRU_tf3 = CNN_GRU_model_tf3.predict(data_tf_3['test'][0])

    '''CNN GRU tf4'''
    CNN_GRU_model_tf4 = KerasModel(inputs=CNN_GRU_model.input, outputs=CNN_GRU_model.output)
    for index, layer in enumerate(CNN_GRU_model_tf4.layers):
        # print(index, layer)
        if not index > 15:
            layer.trainable = False

    CNN_GRU_model_tf4.compile(optimizer=Adam(1e-5), loss=MSE)
    CNN_GRU_time_tf4_start = time.time()
    train(CNN_GRU_model_tf4, data_tf_4, callbacks=None,
          epochs=LSMT_epochs_tf, batch_size=batch_size, verbose=1)
    CNN_GRU_time_tf4_end = time.time()
    CNN_GRU_time_tf4 = CNN_GRU_time_tf4_end - CNN_GRU_time_tf4_start
    y_pred_CNN_GRU_tf4 = CNN_GRU_model_tf4.predict(data_tf_4['test'][0])

    y_pred_CNN_GRU_di1 = CNN_GRU_model.predict(data_tf_1['test'][0])
    y_pred_CNN_GRU_di2 = CNN_GRU_model.predict(data_tf_2['test'][0])
    y_pred_CNN_GRU_di3 = CNN_GRU_model.predict(data_tf_3['test'][0])
    y_pred_CNN_GRU_di4 = CNN_GRU_model.predict(data_tf_4['test'][0])
    # if tf_write_flag:
    with open('CNN_GRU_tf.csv', 'w', newline='') as f:
        csv_writer = csv.writer(f)
        csv_writer.writerow(y_pred_CNN_GRU.flatten())
        csv_writer.writerow(y_pred_CNN_GRU_di1.flatten())
        csv_writer.writerow(y_pred_CNN_GRU_di2.flatten())
        csv_writer.writerow(y_pred_CNN_GRU_di3.flatten())
        csv_writer.writerow(y_pred_CNN_GRU_di4.flatten())
        csv_writer.writerow(y_pred_CNN_GRU_tf1.flatten())
        csv_writer.writerow(y_pred_CNN_GRU_tf2.flatten())
        csv_writer.writerow(y_pred_CNN_GRU_tf3.flatten())
        csv_writer.writerow(y_pred_CNN_GRU_tf4.flatten()) 
        csv_writer.writerow(CNN_GRU_history.history['loss'])

    print('GP-LSTM ' + str(source_Temp) + ' Test RMSE:', rmse_predict)
    print("CNN LSTM RMSE: ", CNN_LSTM_RMSE)
    print("CNN GRU RMSE: ", CNN_GRU_RMSE)

    '''GP TF print'''
    print('GP-LSTM ' + str(source_Temp) + ' train RMSE:', rmse_predict_train)
    print('GP-LSTM ' + str(source_Temp) + ' Test RMSE:', rmse_predict)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_1_Temp) + 'direct RMSE:', rmse_predict_di_1)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_2_Temp) + ' direct RMSE:', rmse_predict_di_2)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_3_Temp) + ' direct RMSE:', rmse_predict_di_3)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_4_Temp) + ' direct RMSE:', rmse_predict_di_4)
    print('GP-LSTM ' + str(source_Temp) + 'to' + str(tf_1_Temp) + ' Transfer_Test RMSE:', rmse_predict_tf_1)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_2_Temp) + ' Transfer_Test RMSE:', rmse_predict_tf_2)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_3_Temp) + ' Transfer_Test RMSE:', rmse_predict_tf_3)
    print('GP-LSTM ' + str(source_Temp) + ' to ' + str(tf_4_Temp) + ' Transfer_Test RMSE:', rmse_predict_tf_4)
    print('train time: ' + str(train_time) + '\nfinetune time: ' + str(fine_time),
          '\n' + str(source_Temp) + 'to' + str(tf_1_Temp) + ' time: ' + str(tf_1_time),
          '\n' + str(source_Temp) + 'to' + str(tf_2_Temp) + ' time: ' + str(tf_2_time),
          '\n' + str(source_Temp) + 'to' + str(tf_3_Temp) + ' time: ' + str(tf_3_time),
          '\n' + str(source_Temp) + 'to' + str(tf_4_Temp) + ' time: ' + str(tf_4_time))

    # plt.close('all')
    plt.figure(figsize=(3.92, 2.36))

    plt.plot(range(len(y_test)), y_test, c='k', lw=2, label='Ture')
    plt.plot(range(len(y_test)), y_pre[0].flatten(), color='r', lw=1, ls='--', marker='o', markevery=5, markersize=3,
              label='CL-GPT', alpha=0.9)
    plt.fill_between(range(len(y_pre[0].flatten())),
                     y_pre[0].flatten() + 2 * np.sqrt(s2[0].flatten()),
                     y_pre[0].flatten() - 2 * np.sqrt(s2[0].flatten()), color='orange', alpha=0.7,
                     label='95% CI')
    
    plt.plot(range(len(y_test)), y_pred_CNN_LSTM.flatten(), color='c', lw=1, ls='--', marker='v', markevery=3, markersize=5,
              label='CNN-LSTM', alpha=0.9)

    plt.plot(range(len(y_test)), y_pred_CNN_GRU.flatten(), color='tab:brown', lw=1, ls='--', marker='X', markevery=5, markersize=3,
             label='CNN-GRU', alpha=0.9)
    plt.xlabel('Times [s]')
    plt.ylabel('SOC')
    plt.tick_params(labelsize=9)

    plt.grid()
    plt.legend(loc="best", fontsize=9)


    """error plot"""
    plt.figure(figsize=(3.92, 2.36))
    plt.subplots_adjust(bottom=0.18, top=0.95)
    plt.subplots_adjust(left=0.135, bottom=0.19, top=0.95)
    plt.plot(range(len(y_test)), y_pre[0].flatten() - y_test, color='r', lw=1, ls='--', marker='o', markevery=5, markersize=3,
             label='CL-GPT', alpha=0.9)

    plt.plot(range(len(y_test)), y_pred_CNN_LSTM.flatten()  - y_test, color='c', lw=1, ls='--', marker='v', markevery=3, markersize=5,
              label='CNN-LSTM', alpha=0.9)
    

    plt.plot(range(len(y_test)), y_pred_CNN_GRU.flatten() - y_test, color='tab:brown', lw=1, ls='--', marker='X', markevery=5,
             markersize=3,
             label='CNN-GRU', alpha=0.9)
    plt.xlabel('Times [s]')
    plt.ylabel('Error')
    plt.tick_params(labelsize=9)

    plt.grid()
    plt.legend(loc="best", fontsize=9)


    print('CNN_LSTM_time: ', CNN_LSTM_time)
    print('CNN_LSTM_time_tf1: ', CNN_LSTM_time_tf1)
    print('CNN_LSTM_time_tf2: ', CNN_LSTM_time_tf2)
    print('CNN_LSTM_time_tf3: ', CNN_LSTM_time_tf3)
    print('CNN_LSTM_time_tf4: ', CNN_LSTM_time_tf4)

    print('CNN_GRU_time: ', CNN_GRU_time)
    print('CNN_GRU_time_tf1: ', CNN_GRU_time_tf1)
    print('CNN_GRU_time_tf2: ', CNN_GRU_time_tf2)
    print('CNN_GRU_time_tf3: ', CNN_GRU_time_tf3)
    print('CNN_GRU_time_tf4: ', CNN_GRU_time_tf4)
    


    write_flag = False
    if write_flag:
        with open('preds.csv','w',newline='') as f:
            csv_writer = csv.writer(f)
            csv_writer.writerow(y_test)
            csv_writer.writerow(y_pre[0].flatten())
            csv_writer.writerow(s2[0].flatten())
            csv_writer.writerow(y_pred_CNN_GRU.flatten())



    plt.show()

if __name__ == '__main__':
    main()
