# encoding: utf-8
"""
@author: 35760
@time: 2020/5/20 17:25
"""
import numpy as np
import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()  # 由于现在是tf2, 使用tf1需要加这个

input_ones_array = np.ones(450)
input_ones_list = input_ones_array.tolist()
input_zeros_array = np.zeros(450)
for i in input_zeros_array:
	input_ones_list.append(i)
input_array = np.array(input_ones_list)
"""
# 输入图像,4D
# shape为 [ batch, in_height, in_weight, in_channel ]
batch:图片的数量
in_height: 图片高度
in_weight: 图片宽度
in_channel 图片的通道数(灰度图该值为1, 彩色图为3)
"""
input_image = input_array.reshape(1, 30, 30, 1)

filter_ones_array = np.ones(3)
filter_ones_list = filter_ones_array.tolist()
filter_zeros_array = np.zeros(3)
for j in filter_zeros_array:
	filter_ones_list.append(j)
for k in filter_ones_array:
	filter_ones_list.append(-k)

filter_array = np.array(filter_ones_list)
filter_image = filter_array.reshape(3, 3, 1, 1)  # 过滤器
print(input_image.shape, filter_image.shape)
op = tf.nn.conv2d(input_image, filter_image, strides=[1, 1, 1, 1], padding='VALID')
print(op)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	print("op:\n", sess.run(op))
