# encoding: utf-8
"""
@author: 35760
@time: 2020/5/20 19:09
tf实现文本的自动生成
"""
import tensorflow as tf
import numpy as np


class DataLoader:
	def __init__(self):
		path = tf.keras.utils.get_file('nietzsche.txt', origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
		with open(path, encoding='utf-8') as f:
			self.raw_text = f.read().lower()
		self.chars = sorted(list(set(self.raw_text)))  # 字符list, 包含英文字母, 阿拉伯数字和标点符号
		self.char_indices = dict((c, i) for i, c in enumerate(self.chars))  # 每个字符以及字符对应的顺序
		self.indices_char = dict((i, c) for i, c in enumerate(self.chars))  # 和char_indices反过来
		self.text = [self.char_indices[c] for c in self.raw_text]  # raw_text每一个字符对应的顺序

	def get_batch(self, seq_length, batch_size):
		seq, next_char = [], []
		for i in range(batch_size):
			index = np.random.randint(0, len(self.text) - seq_length)  # len(self.text)是文本总长度,为600893
			seq.append(self.text[index:index + seq_length])  # 前面40个字符作为输入
			next_char.append(self.text[index + seq_length])  # 下一个字符作为输出
		return np.array(seq), np.array(next_char)  # [batch_size, seq_length], [num_batch]


class RNN(tf.keras.Model):
	def __init__(self, num_chars, batch_size, seq_length):
		super().__init__()
		self.num_chars = num_chars  # txt中不重复的字符个数, 这里是57个
		self.seq_length = seq_length  #
		self.batch_size = batch_size
		# 1.units指的是输出空间的维度
		self.cell = tf.keras.layers.LSTMCell(units=256)  # cell即为网络的架构
		self.dense = tf.keras.layers.Dense(units=self.num_chars)

	def call(self, inputs, from_logits=False):
		# 输入X的尺寸为50*40, 经过one_hot之后,尺寸变为50*40*47,分别是[样本数,序列长度,以及txt中不重复的字符个数]
		inputs = tf.one_hot(inputs, depth=self.num_chars)
		state = self.cell.get_initial_state(batch_size=self.batch_size, dtype=tf.float32)  # 获取一个最初状态
		for t in range(self.seq_length):
			# 1.输入是所有样本在t时刻的one-hot形式,每一个one-hot的长度都是57
			# 2.输出包含了当前t时刻的输出和细胞状态state
			# 3.细胞状态会一直在更新, 由于这里的目标是文本生成,只用取最后一个文本即可
			# 4.下一个时刻只会用上一个时刻的state, 而上一个的时刻输出并不会作为下一个时刻的最后一个输入
			output, state = self.cell(inputs[:, t, :], state)
		logits = self.dense(output)  # 最后一个输出再经过一个全连接层
		if from_logits:
			return logits
		else:
			return tf.nn.softmax(logits)  # 由于目标是

	def predict(self, inputs, temperature=1.):
		batch_size, _ = tf.shape(inputs)
		logits = self(inputs, from_logits=True)
		prob = tf.nn.softmax(logits / temperature).numpy()
		return np.array([np.random.choice(self.num_chars, p=prob[i, :])
						 for i in range(batch_size.numpy())])


def main():

	num_batches = 1000
	seq_length = 40  # 序列的长度
	batch_size = 50  # 样本数
	learning_rate = 1e-3
	data_loader = DataLoader()
	model = RNN(num_chars=len(data_loader.chars), batch_size=batch_size, seq_length=seq_length)
	optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
	for batch_index in range(num_batches):
		X, y = data_loader.get_batch(seq_length, batch_size)  # 输入是文本序列,输出是文本的下一个单词
		with tf.GradientTape() as tape:
			y_pred = model(X)
			loss = tf.keras.losses.sparse_categorical_crossentropy(y_true=y, y_pred=y_pred)
			loss = tf.reduce_mean(loss)
			print("batch %d: loss %f" % (batch_index, loss.numpy()))
		grads = tape.gradient(loss, model.variables)
		optimizer.apply_gradients(grads_and_vars=zip(grads, model.variables))

	# 开始对模型进行评估
	X_, _ = data_loader.get_batch(seq_length, 1)  # 这里只用一个样本即可
	for diversity in [0.2, 0.5, 1.0, 1.2]:
		X = X_
		print("diversity %f:" % diversity)
		for t in range(400):
			y_pred = model.predict(X, diversity)
			print(data_loader.indices_char[y_pred[0]], end='', flush=True)
			X = np.concatenate([X[:, 1:], np.expand_dims(y_pred, axis=1)], axis=-1)
		print("\n")


if __name__ == '__main__':
	main()
