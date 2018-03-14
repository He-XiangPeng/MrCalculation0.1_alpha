# -*- coding: utf-8 -*-
"""
计算分子量的曲线拟合0.2.0.0_alpha,双组拟合
曲线优化视图，添加中文标题，调整字体大小
拟合两条直线
添加图例
改变数据点标记形状，大小
计算两条拟合直线的斜率及截距值
将趋势线添加到曲线中,并修改趋势线的格式，添加R2值
将趋势线外推至y轴
从原始时间数据得到曲线
"""
# Required Packages
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math
from sklearn import datasets, linear_model

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

    Y_parameter.extend([(average(avg_t1)/average(avg_t0))-1, (average(avg_t2)/average(avg_t0))-1,
                       (average(avg_t3)/average(avg_t0)-1)])
    Y_parameter = np.array(Y_parameter)
    # print((np.array(Y_parameter)).shape)
    # print((np.array(X_parameter)).shape)
    Y_parameter = (np.array(Y_parameter) / np.array(X_parameter).reshape(3, )) # reshape（3，）变为一维数组不改变内容
    Z_parameter.extend([math.log(average(avg_t1)/average(avg_t0)), math.log(average(avg_t2)/average(avg_t0)),
                       math.log(average(avg_t3)/average(avg_t0))])
    Z_parameter = (np.array(Z_parameter) / np.array(X_parameter).reshape(3, ))

    return X_parameter, Y_parameter, Z_parameter


# Function to show the resutls of linear fit model
def show_linear_line(X_parameters, Y_parameters, Z_parameters):
    # Create linear regression object
    regr = linear_model.LinearRegression()
    regr1 = linear_model.LinearRegression()
    regr.fit(X_parameters, Y_parameters)
    regr1.fit(X_parameters, Z_parameters)

    # 测试数据类型
    # print(type(regr.predict(X_parameters)))
    # print(type(regr1.predict(X_parameters)))
    # < class 'numpy.ndarray'>
    # < class 'numpy.ndarray'>
    # print(regr1.predict(X_parameters))
    # [9.01061224  9.04836735  9.07102041]

    # Intercept value（截距值）就是θ0的值，coefficient value（系数）就是θ1的值
    # 图内添加趋势线
    # round(regr.coef_[0], 3)数组小数点后保留3位数字
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(111)  # 表示在x*y的网格的格式里，占第z个位置
    ax.text(0.015, 10, '$ \ y_1=$' + str(round(regr.coef_[0], 3)) + 'x' + '+' + str(round(regr.intercept_, 3)),
            fontsize=15)
    ax.text(0.017, 9.9, "$ \ R_1^2=$" + str(round(regr.score(X_parameters, Y_parameters), 4)), fontsize=15)
    ax.text(0.015, 8.8, '$ \ y_2=$' + str(round(regr1.coef_[0], 3)) + 'x' + '+' + str(round(regr1.intercept_, 3)),
            fontsize=15)
    ax.text(0.017, 8.7, "$ \ R_2^2=$" + str(round(regr1.score(X_parameters, Z_parameters), 4)), fontsize=15)
    plt.scatter(X_parameters, Y_parameters, marker='s', color='b', linewidths=2)
    plt.scatter(X_parameters, Z_parameters, marker='p', color='k', linewidths=2)  # k = black


    X_parameters.insert(0, [0])
    # print(X_parameters)
    # [[0], [0.02], [0.015], [0.012]]

    # label : 给所绘制的曲线一个名字，此名字在图示(legend)中显示。
    # 只要在字符串前后添加"$"符号，matplotlib就会使用其内嵌的latex引擎绘制的数学公式。
    # using np.asscalar(a) 将numpy.float64转换为python.float
    plt.plot(X_parameters, regr.predict(X_parameters), label="$ηsp/c$", color='r', linewidth=1.5)
    plt.plot(X_parameters, regr1.predict(X_parameters), label="$lnηr/c$", color='b', linewidth=1.5)

    plt.ylim(8.5, 10.5)
    plt.xlim(0, 0.03)
    plt.xlabel("c(g/ml)")
    plt.title(u"分子量曲线拟合", fontproperties='SimHei', size=25)
    plt.legend()  # 显示图例
    plt.show()




try:
    X, Y, Z = get_data('time.csv')
except OSError as oserr:
    print("文件错误：" + str(oserr))
# print(type(X))
# print(type(Y))
# print(type(Z))
show_linear_line(X, Y, Z)
