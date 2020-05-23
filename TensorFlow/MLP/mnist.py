# encoding: utf-8
"""
@author: 35760
@time: 2020/5/18 21:42
tf实现mnist手写字
"""
import tensorflow as tf
import numpy as np


class MNISTLoader:
	def __init__(self):
		mnist = tf.keras.datasets.mnist  # windows下的文件下载存放在C:\Users\用户名
		# train_data有7万个, test_data有1万个
		(self.train_data, self.train_label), (self.test_data, self.test_label) = mnist.load_data()
		# 1.np.expand_dims(a, axis):  a:数组, axis:添加的数要放置的位置,这里是最后一维
		# 2.MNIST中的图像是灰度图,其值在0~255之间,统一除以255.0相当于做一个MinMaxScaler
		self.train_data = np.expand_dims(self.train_data.astype(np.float32) / 255.0, axis=-1)  # [60000, 28, 28, 1]
		self.test_data = np.expand_dims(self.test_data.astype(np.float32) / 255.0, axis=-1)  # [10000, 28, 28, 1]
		self.train_label = self.train_label.astype(np.int32)  # [60000]
		self.test_label = self.test_label.astype(np.int32)  # [10000]
		self.num_train_data, self.num_test_data = self.train_data.shape[0], self.test_data.shape[0]

	def get_batch(self, batch_size):
		# 从数据集中随机取出batch_size个元素并返回
		index = np.random.randint(0, np.shape(self.train_data)[0], batch_size)
		return self.train_data[index, :], self.train_label[index]


class MLP(tf.keras.Model):
	# 和前面的Linear模型一样,__init__用来架构模型所需的层, call用来描述输入数据如何通过各种层而得到输出
	def __init__(self):
		super().__init__()
		# 1.Flatten层用来将输入"压平",即把多维的输入一维化,常用在从卷积层到全连接层的过渡.
		# 2.Flatten不改变batch的大小, 但是全连接层Dense会改变
		self.flatten = tf.keras.layers.Flatten()
		"""
		Flatten之后的维度是748,怎么通过Dense层变成100的?
		(1)全连接层前后的两层有一个f(W*X+b)的变化,得到加权和之后再通过一个非线性的激活函数,和神经网络的机制一样
		(2)为什么维度发生变化,可以理解为全连接层的神经元个数为100个,它的上一层的神经元个数为748
		"""
		self.dense1 = tf.keras.layers.Dense(units=100, activation=tf.nn.relu)
		self.dense2 = tf.keras.layers.Dense(units=10)  # 默认的激活函数是None, 这也是为什么经常需要两个全连接层的原因?

	def call(self, inputs):  # [batch_size, 28, 28, 1]
		x = self.flatten(inputs)  # [batch_size, 784]
		x = self.dense1(x)  # [batch_size, 100]
		x = self.dense2(x)  # [batch_size, 10]
		output = tf.nn.softmax(x)
		return output


def main():

	# 1.60000条数据,共50个batch,一个batch是1200条数据, 一个epoch是指把50个batch全部跑完的这个过程
	# 2.为什么epoch不为1呢?一个epoch相当于背一次课文,背多了就记住了,但是也会产生过拟合的问题,所以epoch数既不能过大又不能过小
	# 3.前后两个epoch之间的参数是有联系的,并不是孤立的
	num_epochs = 5
	batch_size = 50
	learning_rate = 0.001
	model = MLP()
	data_loader = MNISTLoader()
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	num_batches = int(data_loader.num_train_data // batch_size * num_epochs)  # 可以理解为总的batch数
	for batch_index in range(num_batches):
		X, y = data_loader.get_batch(batch_size)  # 随机选50个数据, 返回这50个数据的输入和输出
		with tf.GradientTape() as tape:
			y_pred = model(X)
			# 1.sparse_categorical_crossentropy交叉熵函数,H(y, y^)= -∑ yi* log(yi^)
			# 2.预测概率分布与真实分布越接近,则交叉熵的值越小,反之则越大.例如H(1, 0.2)>H(1, 0.8)
			loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
			loss = tf.reduce_mean(loss)
			print("batch %d: loss %f" % (batch_index, loss.numpy()))
		grads = tape.gradient(loss, model.variables)
		optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

	# 1.完成6000步迭代之后, 对模型进行评估
	# 2.在评估的时候,需要实例化一个评估器,SparseCategoricalAccuracy可以用于多分类的评估
	sparse_categorical_accuracy = tf.keras.metrics.SparseCategoricalAccuracy()  # 评估器
	num_batches = int(data_loader.num_test_data // batch_size)  # 测试总的batch数:200个
	for batch_index in range(num_batches):
		# 在建模的时候, 采用的是50个输入和50个输出之间建立的模型, 在预测的时候可以直接用50个样本来作为输入
		start_index, end_index = batch_index * batch_size, (batch_index + 1) * batch_size
		y_pred = model.predict(data_loader.test_data[start_index: end_index])  # 取前50个数据
		sparse_categorical_accuracy.update_state(y_true=data_loader.test_label[start_index: end_index], y_pred=y_pred)
	print("test accuracy: %f" % sparse_categorical_accuracy.result())


if __name__ == '__main__':
	main()
