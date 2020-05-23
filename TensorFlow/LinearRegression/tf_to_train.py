# encoding: utf-8
"""
@author: 35760
@time: 2020/5/18 20:16
用TensorFlow实现线性回归,并调用
"""
import tensorflow as tf


class Linear(tf.keras.Model):
    # 1.构造器__init__,初始化模型所需的层
    # 2.模型调用call(),描述输入数据如何通过各种层而得到输出
    def __init__(self):
        super().__init__()
        """
        1.全连接层对输入矩阵A进行f(AW+b)的线性变换 + 激活函数操作,如果没有指定激活函数,就是纯粹的线性变化
        2.实现过程:
        (1)给定输入张量 input = [batch_size, input_dim]
        (2)进行 tf.matmul(input, kernel) + bias 的线性变换
        (3)对线性变换后张量的每个元素通过激活函数 activation
        (4)输出形状为 [batch_size, units] 的二维张量
        (5)units: 输出张量的维度
        """
        self.dense = tf.keras.layers.Dense(
            units=1,  # 输出张量的维度
            activation=None,  # 激活函数, 常用包括 tf.nn.relu, tf.nn.tanh和tf.nn.sigmoid
            kernel_initializer=tf.zeros_initializer(),  # 权重矩阵kernel的初始化器
            bias_initializer=tf.zeros_initializer()  # 偏置向量的初始化器
        )
    #  1.对类的实例Linear进行形如 Linear() 的调用等价于 Linear.__call__()
    #  2.Keras 在模型调用的前后还需要有一些自己的内部操作, 所以暴露出一个专门用于重载的 call() 方法
    #  3.__call__() 中主要调用了 call() 方法, 同时还需要在进行一些 keras 的内部操作

    def call(self, input_data):
        output_data = self.dense(input_data)
        return output_data


def main():

    X = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    y = tf.constant([[10.0], [20.0]])
    model = Linear()
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)
    for i in range(100):
        with tf.GradientTape() as tape:
            # 调用模型 y_pred = model(X) 而不是显式写出 y_pred = a*X + b
            # 由于调用的是线性Linear()模型,这里会自己推断
            y_pred = model(X)
            loss = tf.reduce_mean(tf.square(y_pred - y))
        grads = tape.gradient(loss, model.variables)  # 使用 model.variables 这一属性直接获得模型中的所有变量
        optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))
    print(model.variables)


if __name__ == '__main__':
    main()