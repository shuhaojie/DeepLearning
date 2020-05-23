# encoding: utf-8
"""
@author: 35760
@time: 2020/5/18 16:30
用TensorFlow实现线性回归,并执行梯度下降
"""
import numpy as np
import tensorflow as tf

X_raw = np.array([2013, 2014, 2015, 2016, 2017], dtype=np.float32)
y_raw = np.array([12000, 14000, 15000, 16500, 17500], dtype=np.float32)

X = (X_raw - X_raw.min()) / (X_raw.max() - X_raw.min())
y = (y_raw - y_raw.min()) / (y_raw.max() - y_raw.min())  # 采用最大最小化进行归一化

X = tf.constant(X)  # X和y都是常量
y = tf.constant(y)

a = tf.Variable(initial_value=0.)  # a和b都是变量
b = tf.Variable(initial_value=0.)
variables = [a, b]

num_epoch = 10000
optimizer = tf.keras.optimizers.SGD(learning_rate=5e-4)

"""
TensorFlow 的寻优过程:
1.计算样本的总损失, 写出损失表达式
2.用求导记录器记录每一次的导数grads
3.用apply_gradients根据梯度来更新参数
"""

for e in range(num_epoch):
    with tf.GradientTape() as tape:  # 求导记录器, 记录损失函数的梯度信息
        y_pred = a * X + b
        loss = tf.reduce_sum(tf.square(y_pred - y))  # reduce_sum沿着某个维度求和

    grads = tape.gradient(loss, variables)  # TensorFlow自动计算损失函数关于自变量(模型参数)的梯度

    # TensorFlow自动根据梯度更新参数, grads_and_vars: List of (gradient, variable) pairs.
    # a = [1, 3, 5]， b = [2, 4, 6], 那么zip(a, b) = [(1, 2), (3, 4), ..., (5, 6)]
    optimizer.apply_gradients(grads_and_vars=zip(grads, variables))

print(a, b)


class Linear(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer()
        )

    def call(self, input_data):
        output_data = self.dense(input_data)
        return output_data


model = Linear()
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
for i in range(100):
    with tf.GradientTape() as tape:
        y_pred = model(X)      # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a * X + b
        loss = tf.reduce_mean(tf.square(y_pred - y))
    grads = tape.gradient(loss, model.variables)    # 使用 model.variables 这一属性直接获得模型中的所有变量
    optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
print(model.variables)