# encoding: utf-8
"""
@author: 35760
@time: 2020/5/25 21:24
用python实现AdaBoost算法
"""
from numpy import *
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.font_manager import *


def loadDataSet():  # 加载测试数据
	dataMat = mat([[1., 2.1],
				   [1.5, 1.6],
				   [1.3, 1.],
				   [1., 1.],
				   [2., 1.],
				   [1.1, 1.2]])

	labelList = [1.0, 1.0, -1.0, -1.0, 1.0, 1.0]
	return dataMat, labelList  # 数据集返回的是矩阵类型，标签返回的是列表类型


def stumpClassify(dataMat, dimen, threshVal, threshIneq):  # dimen:第dimen列，也就是第几个特征, threshVal:是阈值  threshIneq：标志
	retArray = ones((shape(dataMat)[0], 1))  # 创造一个 样本数×1 维的array数组
	if threshIneq == 'lt':  # lt表示less than，表示分类方式，对于小于等于阈值的样本点赋值为-1
		retArray[dataMat[:, dimen] <= threshVal] = -1.0
	else:  # 我们确定一个阈值后，有两种分法，一种是小于这个阈值的是正类，大于这个值的是负类，
		# 第二种分法是小于这个值的是负类，大于这个值的是正类，所以才会有这里的if 和else
		retArray[dataMat[:, dimen] > threshVal] = -1.0
	return retArray  # 返回的是一个基分类器的分类好的array数组


def buildStump(dataArr, classLabels, D):
	dataMat = mat(dataArr)
	labelMat = mat(classLabels).T
	m, n = shape(dataMat)
	numStemp = 10
	bestStump = {}
	bestClassEst = mat(zeros((m, 1)))
	minError = inf  # 无穷
	for i in range(n):  # 遍历特征
		rangeMin = dataMat[:, i].min()  # 检查到该特征的最小值
		rangeMax = dataMat[:, i].max()
		stepSize = (rangeMax - rangeMin) / numStemp  # 寻找阈值的步长是最大减最小除以10,你也可以按自己的意愿设置步长公式
		for j in range(-1, int(numStemp) + 1):
			for inequal in ['lt', 'gt']:  # 因为确定一个阈值后，可以有两种分类方式
				threshVal = (rangeMin + float(j) * stepSize)
				predictedVals = stumpClassify(dataMat, i, threshVal,
											  inequal)  # 确定一个阈值后，计算它的分类结果，predictedVals就是基分类器的预测结果，是一个m×1的array数组
				errArr = mat(ones((m, 1)))
				errArr[predictedVals == labelMat] = 0  # 预测值与实际值相同，误差置为0
				weightedEroor = D.T * errArr  # D就是每个样本点的权值，随着迭代，它会变化，这段代码是误差率的公式
				if weightedEroor < minError:  # 选出分类误差最小的基分类器
					minError = weightedEroor  # 保存分类器的分类误差
					bestClassEst = predictedVals.copy()  # 保存分类器分类的结果
					bestStump['dim'] = i  # 保存那个分类器的选择的特征
					bestStump['thresh'] = threshVal  # 保存分类器选择的阈值
					bestStump['ineq'] = inequal  # 保存分类器选择的分类方式
	return bestStump, minError, bestClassEst


def adaBoostTrainDS(dataMat, classLabels, numIt=40):  # 迭代40次，直至误差满足要求，或达到40次迭代
	weakClassArr = []  # 保存每个基分类器的信息，存入列表
	m = shape(dataMat)[0]
	D = mat(ones((m, 1)) / m)
	aggClassEst = mat(zeros((m, 1)))
	for i in range(numIt):
		bestStump, error, classEst = buildStump(dataMat, classLabels, D)
		alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))  # 对应公式 a = 0.5* (1-e)/e
		bestStump['alpha'] = alpha
		weakClassArr.append(bestStump)  # 把每个基分类器存入列表
		print(len(classLabels))
		# print(len(classLabels[0]))
		print(classEst.shape)
		expon = multiply(-1 * alpha * mat(classLabels).T, classEst)  # multiply是对应元素相乘
		D = multiply(D, exp(expon))  # 根据公式 w^m+1 = w^m (e^-a*y^i*G)/Z^m
		D = D / D.sum()  # 归一化
		aggClassEst += alpha * classEst  # 分类函数 f(x) = a1 * G1
		aggErrors = multiply(sign(aggClassEst) != mat(classLabels).T, ones((m, 1)))  # 分错的矩阵
		errorRate = aggErrors.sum() / m  # 分错的个数除以总数，就是分类误差率
		if errorRate == 0.0:  # 误差率满足要求，则break退出
			break
	return weakClassArr, aggClassEst


def adaClassify(datToClass, classifierArr):  # 预测分类
	dataMat = mat(datToClass)  # 测试数据集转为矩阵格式
	m = shape(dataMat)[0]
	aggClassEst = mat(zeros((m, 1)))
	for i in range(len(classifierArr)):
		classEst = stumpClassify(dataMat, classifierArr[i]['dim'], classifierArr[i]['thresh'],
								 classifierArr[i]['ineq'])  # 可以对比文章开头的图，其实就是那个公式
		aggClassEst += classifierArr[i]['alpha'] * classEst
	return sign(aggClassEst)


def draw_figure(dataMat, labelList, weakClassArr):  # 画图
	# myfont = FontProperties(fname='/usr/share/fonts/simhei.ttf')    # 显示中文
	matplotlib.rcParams['axes.unicode_minus'] = False  # 防止坐标轴的‘-’变为方块
	matplotlib.rcParams["font.sans-serif"] = ["simhei"]  # 第二种显示中文的方法
	fig = plt.figure()  # 创建画布
	ax = fig.add_subplot(111)  # 添加子图

	red_points_x = []  # 红点的x坐标
	red_points_y = []  # 红点的y坐标
	blue_points_x = []  # 蓝点的x坐标
	blue_points_y = []  # 蓝点的y坐标
	m, n = shape(dataMat)  # 训练集的维度是 m×n ，m就是样本个数，n就是每个样本的特征数
	dataSet_list = array(dataMat)  # 训练集转化为array数组

	for i in range(m):  # 遍历训练集，把红点，蓝点分开存入
		if labelList[i] == 1:
			red_points_x.append(dataSet_list[i][0])  # 红点x坐标
			red_points_y.append(dataSet_list[i][1])
		else:
			blue_points_x.append(dataSet_list[i][0])
			blue_points_y.append(dataSet_list[i][1])

	line_thresh = 0.025  # 画线阈值，就是不要把线画在点上，而是把线稍微偏移一下，目的就是为了让图更加美观直接
	annotagte_thresh = 0.03  # 箭头间隔，也是为了美观
	x_min = y_min = 0.50  # 自设的坐标显示的最大最小值，这里固定死了，应该是根据训练集的具体情况设定
	x_max = y_max = 2.50

	v_line_list = []  # 把竖线阈值的信息存起来，包括阈值大小，分类方式，alpha大小都存起来
	h_line_list = []  # 横线阈值也是如此，因为填充每个区域时，竖阈值和横阈值是填充边界，是不一样的，需各自分开存贮
	for baseClassifier in weakClassArr:  # 画阈值
		if baseClassifier['dim'] == 0:  # 画竖线阈值
			if baseClassifier['ineq'] == 'lt':  # 根据分类方式,lt时
				ax1 = ax.vlines(baseClassifier['thresh'] + line_thresh, y_min, y_max, colors='green', label='阈值')  # 画直线
				ax.arrow(baseClassifier['thresh'] + line_thresh, 1.5, 0.08, 0, head_width=0.05,
						 head_length=0.02)  # 显示箭头
				ax.text(baseClassifier['thresh'] + annotagte_thresh, 1.5 + line_thresh,
						str(round(baseClassifier['alpha'], 2)))  # 画alpha值
				v_line_list.append(
					[baseClassifier['thresh'], 1, baseClassifier['alpha']])  # 把竖线信息存入，注意分类方式，lt就存1,gt就存-1

			else:  # gt时，分类方式不同，箭头指向也不同
				ax.vlines(baseClassifier['thresh'] + line_thresh, y_min, y_max, colors='green', label="阈值")
				ax.arrow(baseClassifier['thresh'] + line_thresh, 1., -0.08, 0, head_width=0.05, head_length=0.02)
				ax.text(baseClassifier['thresh'] + annotagte_thresh, 1. + line_thresh,
						str(round(baseClassifier['alpha'], 2)))
				v_line_list.append([baseClassifier['thresh'], -1, baseClassifier['alpha']])
		else:  # 画横线阈值
			if baseClassifier['ineq'] == 'lt':  # 根据分类方式，lt时
				ax.hlines(baseClassifier['thresh'] + line_thresh, x_min, x_max, colors='black', label="阈值")
				ax.arrow(1.5 + line_thresh, baseClassifier['thresh'] + line_thresh, 0., 0.08, head_width=0.05,
						 head_length=0.05)
				ax.text(1.5 + annotagte_thresh, baseClassifier['thresh'] + 0.04, str(round(baseClassifier['alpha'], 2)))
				h_line_list.append([baseClassifier['thresh'], 1, baseClassifier['alpha']])
			else:  # gt时
				ax.hlines(baseClassifier['thresh'] + line_thresh, x_min, x_max, colors='black', label="阈值")
				ax.arrow(1.0 + line_thresh, baseClassifier['thresh'], 0., 0.08, head_width=-0.05, head_length=0.05)
				ax.text(1.0 + annotagte_thresh, baseClassifier['thresh'] + 0.04, str(round(baseClassifier['alpha'], 2)))
				h_line_list.append([baseClassifier['thresh'], -1, baseClassifier['alpha']])
	v_line_list.sort(key=lambda x: x[0])  # 我们把存好的竖线信息按照阈值大小从小到大排序，因为我们填充颜色是从左上角开始，所以竖线从小到大排
	h_line_list.sort(key=lambda x: x[0], reverse=True)  # 横线从大到小排序
	v_line_list_size = len(v_line_list)  # 排好之后，得到竖线有多少条
	h_line_list_size = len(h_line_list)  # 得到横线有多少条
	alpha_value = [x[2] for x in v_line_list] + [y[2] for y in h_line_list]  # 把属性横线的所有alpha值取出来，这里也证实了上面的排序不是无用功

	for i in range(h_line_list_size + 1):  # 开始填充颜色，(横线的条数+1) × (竖线的条数+1) = 分割的区域数,然后开始往这几个区域填颜色
		for j in range(v_line_list_size + 1):  # 我们是左上角开始填充直到右下角，所以采用这种遍历方式
			list_test = list(multiply([1] * j + [-1] * (v_line_list_size - j), [x[1] for x in v_line_list])) + list(
				multiply([-1] * i + [1] * (h_line_list_size - i), [x[1] for x in h_line_list]))
			# 上面是一个规律公式，后面会用文字解释它
			temp_value = multiply(alpha_value,
								  list_test)  # list_test其实就是加减号，我们知道了所有alpha值，可是每个alpha是相加还是相加，这就是list_test作用了
			reslut_test = sign(sum(temp_value))  # 计算完后，sign一下，然后根据结果进行分类
			if reslut_test == 1:  # 如果是1,就是正类红点
				color_select = 'orange'  # 填充的颜色是橘红色
				hatch_select = '.'  # 填充图案是。
			# print("是正类，红点")
			else:  # 如果是-1,那么是负类蓝点
				color_select = 'green'  # 填充的颜色是绿色
				hatch_select = '*'  # 填充图案是*
			# print("是负类，蓝点")
			if i == 0:  # 上边界     现在开始填充了，用fill_between函数，我们需要得到填充的x坐标范围，和y的坐标范围，x范围就是多条竖线阈值夹着的区域，y范围是横线阈值夹着的范围
				if j == 0:  # 左上角
					ax.fill_between(x=[x for x in arange(x_min, v_line_list[j][0] + line_thresh, 0.001)], y1=y_max,
									y2=h_line_list[i][0] + line_thresh, color=color_select, alpha=0.3,
									hatch=hatch_select)
				elif j == v_line_list_size:  # 右上角
					ax.fill_between(x=[x for x in arange(v_line_list[-1][0] + line_thresh, x_max, 0.001)], y1=y_max,
									y2=h_line_list[i][0] + line_thresh, color=color_select, alpha=0.3,
									hatch=hatch_select)
				else:  # 中间部分
					ax.fill_between(x=[x for x in
									   arange(v_line_list[j - 1][0] + line_thresh, v_line_list[j][0] + line_thresh,
											  0.001)], y1=y_max, y2=h_line_list[i][0] + line_thresh, color=color_select,
									alpha=0.3, hatch=hatch_select)
			elif i == h_line_list_size:  # 下边界
				if j == 0:  # 左下角
					ax.fill_between(x=[x for x in arange(x_min, v_line_list[j][0] + line_thresh, 0.001)],
									y1=h_line_list[-1][0] + line_thresh, y2=y_min, color=color_select, alpha=0.3,
									hatch=hatch_select)
				elif j == v_line_list_size:  # 右下角
					ax.fill_between(x=[x for x in arange(v_line_list[-1][0] + line_thresh, x_max, 0.001)],
									y1=h_line_list[-1][0] + line_thresh, y2=y_min, color=color_select, alpha=0.3,
									hatch=hatch_select)
				else:  # 中间部分
					ax.fill_between(x=[x for x in
									   arange(v_line_list[j - 1][0] + line_thresh, v_line_list[j][0] + line_thresh,
											  0.001)], y1=h_line_list[-1][0] + line_thresh, y2=y_min,
									color=color_select, alpha=0.3, hatch=hatch_select)
			else:
				if j == 0:  # 中左角
					ax.fill_between(x=[x for x in arange(x_min, v_line_list[j][0] + line_thresh, 0.001)],
									y1=h_line_list[i - 1][0] + line_thresh, y2=h_line_list[i][0] + line_thresh,
									color=color_select, alpha=0.3, hatch=hatch_select)
				elif j == v_line_list_size:  # 中右角
					ax.fill_between(x=[x for x in arange(v_line_list[-1][0] + line_thresh, x_max, 0.001)],
									y1=h_line_list[i - 1][0] + line_thresh, y2=h_line_list[i][0] + line_thresh,
									color=color_select, alpha=0.3, hatch=hatch_select)
				else:  # 中间部分
					ax.fill_between(x=[x for x in
									   arange(v_line_list[j - 1][0] + line_thresh, v_line_list[j][0] + line_thresh,
											  0.001)], y1=h_line_list[i - 1][0] + line_thresh,
									y2=h_line_list[i][0] + line_thresh, color=color_select, alpha=0.3,
									hatch=hatch_select)

	ax.scatter(red_points_x, red_points_y, s=30, c='red', marker='s', label="red points")  # 画红点
	ax.scatter(blue_points_x, blue_points_y, s=40, label="blue points")  # 画蓝点
	ax.set_xlabel("x")
	ax.set_ylabel("y")
	ax.legend()  # 显示图例    如果你想用legend设置中文字体，参数设置为 prop=myfont
	ax.set_title("图 5 AdaBoost分类", position=(0.5, -0.175))  # 设置标题，改变位置，可以放在图下面，这个position是相对于图片的位置
	plt.show()


if __name__ == '__main__':  # 运行函数
	dataMat, labelList = loadDataSet()  # 加载数据集
	print(dataMat.shape)
	weakClassArr, aggClassEst = adaBoostTrainDS(dataMat, labelList)
	draw_figure(dataMat, labelList, weakClassArr)  # 画图
	print('weakClassArr', weakClassArr)
	print('aggClassEst', aggClassEst)
	classify_result = adaClassify([0.7, 1.7], weakClassArr)  # 预测的分类结果，测试集我们用的是[0.7,1.7]测试集随便选
	print("结果是:", classify_result)
