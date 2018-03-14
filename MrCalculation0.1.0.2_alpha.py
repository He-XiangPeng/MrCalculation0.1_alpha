# -*- coding: utf-8 -*-
"""
计算分子量的曲线拟合0.1.0.2_alpha,双组拟合
曲线优化视图，添加中文标题，调整字体大小
拟合两条直线
"""
# Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


# Function to get data X, Y , Z
def get_data(file_name):
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    Z_parameter = []
    for single_sample_concentration, single_np, single_nr in zip(data['sample_concentration'], data['np'], data['nr']):
        X_parameter.append([float(single_sample_concentration)])
        Y_parameter.append(float(single_np))
        Z_parameter.append(float(single_nr))
    return X_parameter, Y_parameter, Z_parameter


# Function to show the resutls of linear fit model
def show_linear_line(X_parameters, Y_parameters, Z_parameters):
    # Create linear regression object


    regr = linear_model.LinearRegression()
    regr1 = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)

    regr1.fit(X_parameters, Z_parameters)
    plt.figure(figsize=(12, 8))
    plt.scatter(X_parameters, Y_parameters, color='b')
    plt.scatter(X_parameters, Z_parameters, color='k')
    plt.plot(X_parameters, regr.predict(X_parameters), color='r', linewidth=1)
    plt.plot(X_parameters, regr1.predict(X_parameters), color='b', linewidth=1)

    # # Explained variance score: 1 is perfect prediction
    # print('Variance score y1: %.4f' % regr.score(X_parameters, Y_parameters))
    # print('Variance score y2: %.4f' % regr1.score(X_parameters, Z_parameters))

    plt.ylim(8.5, 10.5)
    plt.xlim(0, 0.03)
    plt.xlabel("c")
    plt.title(u"分子量曲线拟合", fontproperties='SimHei',size=20)

    plt.show()


X, Y, Z = get_data('rawdata.csv')
show_linear_line(X, Y, Z)
