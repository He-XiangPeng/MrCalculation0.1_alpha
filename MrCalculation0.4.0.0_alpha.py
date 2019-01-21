# -*- coding: utf-8 -*-
"""
计算分子量的曲线拟合0.4.0.0_alpha
曲线优化视图，添加中文标题，调整字体大小
拟合两条直线
添加图例
改变数据点标记形状，大小
计算两条拟合直线的斜率及截距值
将趋势线添加到曲线中,并修改趋势线的格式，添加R2值
将趋势线外推至y轴
从最初原始时间数据data.csv得到曲线
将分子量的值显示在图中
代码优化
"""
# Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn import linear_model


# 数组求平均数
def average(seq):
    return float(sum(seq)) / len(seq)


# Function to get data X, Y , Z
def get_data(file_name):
    data = pd.read_csv(file_name)
    X_parameter = []
    Y_parameter = []
    Z_parameter = []

    # avg_t0 = []
    # avg_t1 = []
    # avg_t2 = []
    # avg_t3 = []
    # for t0, t1, t2, t3 in zip(data['t0'], data['t1'],
    #   data['t2'], data['t3']):
    #     avg_t0.append(float(t0))
    #     avg_t1.append(float(t1))
    #     avg_t2.append(float(t2))
    #     avg_t3.append(float(t3))
    avg_t0 = [float(t0) for t0 in data['t0']]
    avg_t1 = [float(t1) for t1 in data['t1']]
    avg_t2 = [float(t2) for t2 in data['t2']]
    avg_t3 = [float(t3) for t3 in data['t3']]

    X_parameter.extend([[data['weight'][0] / 30],
                        [data['weight'][0] / 30 * 0.75],
                        [data['weight'][0] / 30 * 0.6]])
    Y_parameter.extend([(average(avg_t1) / average(avg_t0)) - 1,
                        (average(avg_t2) / average(avg_t0)) - 1,
                        (average(avg_t3) / average(avg_t0) - 1)])
    Y_parameter = np.array(Y_parameter)
    Y_parameter = (np.array(Y_parameter) / np.array(X_parameter).reshape(3, ))
    Z_parameter.extend([math.log(average(avg_t1) / average(avg_t0)),
                        math.log(average(avg_t2) / average(avg_t0)),
                        math.log(average(avg_t3) / average(avg_t0))])
    Z_parameter = (np.array(Z_parameter) / np.array(X_parameter).reshape(3, ))
    return X_parameter, Y_parameter, Z_parameter


# Function to show the resutls of linear fit model
def show_linear_line(X_parameters, Y_parameters, Z_parameters):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr1 = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    regr1.fit(X_parameters, Z_parameters)

    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)
    ax.text(0.015, 10, r'$ \ y_1=$' + str(round(regr.coef_[0], 4)) +
            'x' + '+' +
            str(round(regr.intercept_, 4)), fontsize=15)
    ax.text(0.017, 9.9, r"$ \ R_1^2=$" +
            str(round(regr.score(X_parameters, Y_parameters), 4)), fontsize=15)
    ax.text(0.015, 8.8, r'$ \ y_2=$' +
            str(round(regr1.coef_[0], 4)) + 'x' + '+' +
            str(round(regr1.intercept_, 4)),
            fontsize=15)
    ax.text(0.017, 8.7, r"$ \ R_2^2=$" +
            str(round(regr1.score(X_parameters, Z_parameters), 4)),
            fontsize=15)
    ax.text(0.001, 10, r'$ \bar M=(\frac{[\eta]}{K})^{1/\alpha}$=' +
            str(round(math.pow((regr1.intercept_ * regr.coef_[0] -
                                regr.intercept_ * regr1.coef_[0]) /
                               (regr.coef_[0] - regr1.coef_[0]) /
                               1.166 * 100, 1 / 0.871), 2)), fontsize=20)
    plt.scatter(X_parameters, Y_parameters,
                marker='s', color='b', linewidths=2)
    plt.scatter(X_parameters, Z_parameters,
                marker='p', color='k', linewidths=2)

    X_parameters.insert(0, [0])
    plt.plot(X_parameters, regr.predict(X_parameters),
             label="$\eta_{sp}/c$", color='r', linewidth=1.2)
    plt.plot(X_parameters, regr1.predict(X_parameters),
             label="$ln \eta_r/c$", color='b', linewidth=1.2)
    plt.ylim(8.5, 10.5)
    plt.xlim(0, 0.03)
    plt.xlabel("c(g/ml)", color='r', fontsize=20)
    plt.title("Molecular weight curve fitting", size=20)
    plt.legend()
    plt.show()


try:
    X, Y, Z = get_data('data.csv')
except OSError as oserr:
    print("文件错误：" + str(oserr))
show_linear_line(X, Y, Z)
