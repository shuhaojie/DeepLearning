# encoding: utf-8
"""
@author: 35760
@time: 2020/5/22 21:46
CART树的python实现
"""

from numpy import *
import pandas as pd
import sys
sys.path.append(r"C:\Users\35760\PycharmProjects\DeepLearning\DecisionTree\Package")  # 如果是同级目录下的调用,可以不要这个
import PlotTrees


def calc_gini(data_set):
	num_entries = len(data_set)
	label_counts = {}
	for feat_vec in data_set:
		current_label = feat_vec[-1]
		if current_label not in label_counts.keys():
			label_counts[current_label] = 0
			label_counts[current_label] += 1
	gini = 1.0
	for label in label_counts:
		prob = float(label_counts[label]) / num_entries
		gini -= prob * prob
	return gini


# 对离散变量划分数据集，取出该特征取值为value的所有样本
def split_dataset(data_set, axis, values):
	ret_dataset = []
	for feat_vec in data_set:
		if feat_vec[axis] == values:
			reduced_featvec = feat_vec[:axis]
			reduced_featvec.extend(feat_vec[axis + 1:])
			ret_dataset.append(reduced_featvec)
	return ret_dataset


def split_continuous_dataset(dataset, axis, values, direction):
	ret_dataset = []

	for feat_vec in dataset:
		if direction == 0:
			if feat_vec[axis] > values:
				reduced_feat_vec = feat_vec[:axis]
				reduced_feat_vec.extend(feat_vec[axis + 1:])
				ret_dataset.append(reduced_feat_vec)
		else:
			if feat_vec[axis] <= values:
				reduced_feat_vec = feat_vec[:axis]
				reduced_feat_vec.extend(feat_vec[axis + 1:])
				ret_dataset.append(reduced_feat_vec)
	return ret_dataset


def choose_best_feature_to_split(data_set, labels):
	num_features = len(data_set[0]) - 1  # 特征的个数,最后一个是标签
	best_gini_index = 100000.0
	best_feature = -1
	best_split_dict = {}
	for i in range(num_features):
		feat_list = [example[i] for example in data_set]  # 每一个样本的第i个特征
		if type(feat_list[0]).__name__ == 'float' or type(feat_list[0]).__name__ == 'int':  # 如果特征是数值型的
			sort_feat_list = sorted(feat_list)
			split_list = []
			for j in range(len(sort_feat_list) - 1):
				split_list.append((sort_feat_list[j] + sort_feat_list[j+1])/2.0)
			best_split_gini = 10000
			slen = len(split_list)
			for j in range(slen):
				value = split_list[j]
				new_gini_index = 0.0
				sub_data_set0 = split_continuous_dataset(data_set, i, value, 0)
				sub_data_set1 = split_continuous_dataset(data_set, i, value, 1)
				prob0 = len(sub_data_set0) / float(len(data_set))
				new_gini_index += prob0 * calc_gini(sub_data_set0)
				prob1 = len(sub_data_set1) / float(len(data_set))
				new_gini_index += prob1 * calc_gini(sub_data_set1)
				if new_gini_index < best_split_gini:
					best_split_gini = new_gini_index
					best_split = j
			best_split_dict[labels[i]] = split_list[best_split]
			gini_index = best_split_gini
		else:
			unique_vals = set(feat_list)
			new_gini_index = 0.0
			for value in unique_vals:
				sub_data_set = split_dataset(data_set, i, value)
				prob = len(sub_data_set) / float(len(data_set))
				new_gini_index += prob * calc_gini(sub_data_set)
			gini_index = new_gini_index
		if gini_index < best_gini_index:
			best_gini_index = gini_index
			best_feature = i
	if type(data_set[0][best_feature]).__name__ == 'float' or type(data_set[0][best_feature]).__name__ == 'int':
		best_split_value = best_split_dict[labels[best_feature]]
		labels[best_feature] = labels[best_feature] + '<=' + str(best_split_value)
		for i in range(shape(data_set)[0]):
			if data_set[i][best_feature] <= best_split_value:
				data_set[i][best_feature] = 1
			else:
				data_set[i][best_feature] = 0
	return best_feature


def majority_cnt(class_list):
	class_count = {}
	for vote in class_list:
		if vote not in class_count.keys():
			class_count[vote] = 0
		class_count[vote] += 1
	return max(class_count)


# 主程序，递归产生决策树
def create_tree(data_set, labels, data_full, labels_full):
	class_list = [example[-1] for example in data_set]
	if class_list.count(class_list[0]) == len(class_list):
		return class_list[0]
	if len(data_set[0]) == 1:
		return majority_cnt(class_list)
	best_feat = choose_best_feature_to_split(data_set, labels)
	best_feat_label = labels[best_feat]
	myTree = {best_feat_label: {}}
	feat_values = [example[best_feat] for example in data_set]
	unique_vals = set(feat_values)
	if type(data_set[0][best_feat]).__name__ == 'str':
		current_label = labels_full.index(labels[best_feat])
		feat_values_full = [example[current_label] for example in data_full]
		unique_vals_full = set(feat_values_full)
	del (labels[best_feat])
	for value in unique_vals:
		sub_labels = labels[:]
		if type(data_set[0][best_feat]).__name__ == 'str':
			unique_vals_full.remove(value)
		myTree[best_feat_label][value] = create_tree(split_dataset(data_set, best_feat, value),
													 sub_labels, data_full, labels_full)
	if type(data_set[0][best_feat]).__name__ == 'str':
		for value in unique_vals_full:
			myTree[best_feat_label][value] = majority_cnt(class_list)
	return myTree


def main():
	df = pd.read_csv(r'.\Data\watermelon.csv')
	data = df.values[:11, 1:].tolist()
	data_full = data[:]
	labels = df.columns.values[1:-1].tolist()
	labels_full = labels[:]
	myTree = create_tree(data, labels, data_full, labels_full)

	PlotTrees.createPlot(myTree)


if __name__ == '__main__':
	main()
