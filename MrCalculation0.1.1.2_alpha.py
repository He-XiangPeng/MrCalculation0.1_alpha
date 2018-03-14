# -*- coding: utf-8 -*-
"""
计算分子量的曲线拟合0.1.1.2_alpha,双组拟合
曲线优化视图，添加中文标题，调整字体大小
拟合两条直线
添加图例
改变数据点标记形状，大小
计算两条拟合直线的斜率及截距值
将趋势线添加到曲线中,并修改趋势线的格式
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

    # Intercept value（截距值）就是θ0的值，coefficient value（系数）就是θ1的值
    # linear_plot_parameters = {}
    # linear_plot_parameters['intercept'] = regr.intercept_
    # linear_plot_parameters['coefficient'] = regr.coef_
    # linear_plot_parameters_1 = {}
    # linear_plot_parameters_1['intercept'] = regr1.intercept_
    # linear_plot_parameters_1['coefficient'] = regr1.coef_
    # print(linear_plot_parameters)
    # print(linear_plot_parameters_1)
    # Explained variance score: 1 is perfect prediction
    # print('Variance score y1: %.4f' % regr.score(X_parameters, Y_parameters))
    # print('Variance score y2: %.4f' % regr1.score(X_parameters, Z_parameters))



    # 图内添加趋势线
    # round(regr.coef_[0], 3)数组小数点后保留3位数字

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111)#表示在x*y的网格的格式里，占第z个位置
    ax.text(0.015, 9.75, 'y=' + str(round(regr.coef_[0], 3)) + 'x' + '+' + str(round(regr.intercept_, 3)), fontsize=15)
    ax.text(0.015, 8.95, 'y=' + str(round(regr1.coef_[0], 3)) + 'x' + '+' + str(round(regr1.intercept_, 3)), fontsize=15)

    plt.scatter(X_parameters, Y_parameters, marker='s', color='b', linewidths=2.5)
    plt.scatter(X_parameters, Z_parameters, marker='p', color='k', linewidths=2.5)  # k = black

    # label : 给所绘制的曲线一个名字，此名字在图示(legend)中显示。
    # 只要在字符串前后添加"$"符号，matplotlib就会使用其内嵌的latex引擎绘制的数学公式。
    plt.plot(X_parameters, regr.predict(X_parameters), label="$ηsp/c$", color='r', linewidth=1.8)
    plt.plot(X_parameters, regr1.predict(X_parameters), label="$lnηr/c$", color='b', linewidth=1.8)



    plt.ylim(8.5, 10.5)
    plt.xlim(0, 0.03)
    plt.xlabel("c(g/ml)")
    plt.title(u"分子量曲线拟合", fontproperties='SimHei', size=25)
    plt.legend()  # 显示图例
    plt.show()


X, Y, Z = get_data('rawdata.csv')
show_linear_line(X, Y, Z)
