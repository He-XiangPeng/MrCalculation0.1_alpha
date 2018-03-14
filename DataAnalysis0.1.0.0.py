# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math

# 数组求平均数


def average(seq):
    return float(sum(seq)) / len(seq)

# Function to get data X, Y , Z


def get_data(file_name):
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    Z_parameter = []
    avg_t0 = []
    avg_t1 = []
    avg_t2 = []
    avg_t3 = []
    for single_sample_concentration, t0, t1, t2, t3 in zip(data['sample_concentration'], data['t0'], data['t1'], data['t2'], data['t3']):
        X_parameter.append([float(single_sample_concentration)])
        avg_t0.append(float(t0))
        avg_t1.append(float(t1))
        avg_t2.append(float(t2))
        avg_t3.append(float(t3))

    Y_parameter.extend([(average(avg_t1) / average(avg_t0)) - 1, (average(avg_t2) / average(avg_t0)) - 1,
                        (average(avg_t3) / average(avg_t0) - 1)])
    Y_parameter = np.array(Y_parameter)
    # print((np.array(Y_parameter)).shape)
    # print((np.array(X_parameter)).shape)
    # reshape（3，）变为一维数组不改变内容
    Y_parameter = (np.array(Y_parameter) / np.array(X_parameter).reshape(3, ))
    Z_parameter.extend([math.log(average(avg_t1) / average(avg_t0)), math.log(average(avg_t2) / average(avg_t0)),
                        math.log(average(avg_t3) / average(avg_t0))])
    Z_parameter = (np.array(Z_parameter) / np.array(X_parameter).reshape(3, ))

    return X_parameter, Y_parameter, Z_parameter


try:
    X, Y, Z = get_data('time.csv')
except OSError as oserr:
    print("文件错误：" + str(oserr))

print(X)
print(Y)
print(Z)
