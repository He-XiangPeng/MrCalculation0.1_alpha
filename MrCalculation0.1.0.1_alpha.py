# -*- coding: utf-8 -*-
"""
计算分子量的曲线拟合0.1.0.1_alpha,双组拟合
曲线优化视图，添加中文标题
只能拟合一条直线
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


# 测试get_data方法
# print(get_data("rawdata.csv"))

# 定义交换变量函数
def swap(a, b):
    t = a
    a = b
    b = t


# Function to show the resutls of linear fit model
def show_linear_line(X_parameters, Y_parameters, Z_parameters):
    # Create linear regression object

    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.figure(figsize=(12, 8))
    plt.scatter(X_parameters, Y_parameters, color='b')
    plt.scatter(X_parameters, Z_parameters, color='k')
    plt.plot(X_parameters, regr.predict(X_parameters), color='r', linewidth=2)
    # plt.plot(X_parameters, regr.predict(X_parameters), color='r', linewidth=2)
    plt.ylim(8.5, 10.5)
    plt.xlim(0, 0.03)
    plt.xlabel("c")
    plt.title(u"分子量曲线拟合", fontproperties='SimHei')

    plt.show()


X, Y, Z = get_data('rawdata.csv')
show_linear_line(X, Y, Z)
