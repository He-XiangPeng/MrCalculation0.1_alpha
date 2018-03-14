# -*- coding: utf-8 -*-
"""
计算分子量的曲线拟合0.1alpha,单组拟合
"""
# Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import datasets, linear_model


# Function to get data
def get_data(file_name):
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    for single_sample_concentration, single_np in zip(data['sample_concentration'], data['np']):
        X_parameter.append([float(single_sample_concentration)])
        Y_parameter.append(float(single_np))
    return X_parameter, Y_parameter


# 测试get_data方法
# print(get_data("data.csv"))

# Function to show the resutls of linear fit model
def show_linear_line(X_parameters, Y_parameters):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    plt.scatter(X_parameters, Y_parameters, color='blue')
    plt.plot(X_parameters, regr.predict(X_parameters), color='red', linewidth=2)
    plt.xticks(())
    plt.yticks(())
    plt.xlabel("c")
    # plt.title("Predicted value of House")

    plt.show()


X, Y = get_data('rawdata.csv')
show_linear_line(X, Y)
