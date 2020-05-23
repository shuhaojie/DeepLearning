# encoding: utf-8
"""
@author: 35760
@time: 2020/5/18 16:18
用numpy实现线性回归的梯度下降
"""
import numpy as np

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())  # 采用最大最小化进行归一化

a, b = 0, 0

num_epoch = 10000  # 遍历次数
learning_rate = 5e-4
for e in range(num_epoch):
    # 手动计算损失函数关于自变量（模型参数）的梯度
    y_pred = a * X + b
    grad_a, grad_b = 2 * (y_pred - y).dot(X), 2 * (y_pred - y).sum()  # 梯度需要手工求导得到

    # 更新参数
    a, b = a - learning_rate * grad_a, b - learning_rate * grad_b  # 最基础的参数更新方法:梯度下降

print(a, b)