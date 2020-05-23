# encoding: utf-8
"""
@author: 35760
@time: 2020/5/14 21:54
用
"""
from numpy.random import RandomState
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()  # 由于现在是tf2, 使用tf1需要加这个


batch_size = 8
"""
1.tf.random_normal([2, 3]生成一个2*3的矩阵,矩阵的标准差为1,均值为0(默认),seed=1表示每次生成的随机数都一样
2.tf.Variable:变量的声明函数,通过创建Variable类的实例向graph中添加变量
  Variable()需要初始值,一旦初始值确定,那么该变量的类型和形状都确定了
3.w1和w2都是神经网络的参数
"""
w1 = tf.Variable(tf.random_normal([2, 3], stddev=1, seed=1))
w2 = tf.Variable(tf.random_normal([3, 1], stddev=1, seed=1))

"""
1.tf.placeholder:定义一个位置,这个位置中的数据在程序运行时再指定,这样程序就不需要生成大量的常量来输入数据,而只需要将数据通过
  placeholder传入tensorflow计算图
2.placeholder的类型和其他张量一样,定义好了就不能修改
"""
x = tf.placeholder(tf.float32, shape=(None, 2), name="x-input")
y_ = tf.placeholder(tf.float32, shape=(None, 1), name='y-input')  # y_是真实值

a = tf.matmul(x, w1)
y = tf.matmul(a, w2)  # tf.matmul:将矩阵a乘以矩阵b,生成a*b,前向传播的过程

y = tf.sigmoid(y)  # 这个地方其实就是一个神经网络的形式了,y是预测值

# ===============================================================================================
# 1.tf.log:计算TensorFlow的自然对数
# 2.tf.clip_by_value(A, min, max):输入一个张量A,把A中的每一个元素的值都压缩在min和max之间
#   小于min的让它等于min，大于max的元素的值等于max
# 3.tf.reduce_mean:用于计算张量tensor沿着指定的数轴上的的平均值,主要用作降维或者计算tensor(图像)的平均值
# 4.这句用来定义损失函数,这个损失函数叫交叉熵,是tensorflow中一种常见的损失函数,和神经网络中的损失函数
# def mse_loss(y_true, y_pre):
#   return ((y_true - y_pre) ** 2).mean()
# 不太一样
# ===============================================================================================
cross_entropy = -tf.reduce_mean(y_ * tf.log(tf.clip_by_value(y, 1e-10, 1.0))
                                + (1 - y_) * tf.log(tf.clip_by_value(1 - y, 1e-10, 1.0)))
print('cross_entropy:', cross_entropy)  # >>>cross_entropy: Tensor("Neg:0", shape=(), dtype=float32)
# ===============================================================================================
# 1. tf.train.AdamOptimizer:此函数是Adam优化算法,是一个寻找全局最优点的优化算法
# 2. tf.train.Optimizer.minimize(loss), loss:最小化的目标
# ===============================================================================================
train_step = tf.train.AdamOptimizer(0.001).minimize(cross_entropy)

rdm = RandomState(1)  # 这里的1为随机数种子,只要随机数种子seed相同,产生的随机数系列就相同
dataset_size = 128
X = rdm.rand(128, 2)
# ===============================================================================================
# 1. 注意在算法中一般用Y来表示标签
# 2. 这里所有的x1+x2 < 1视为正样本(比如零件合格),其他视为负样本(零件不合格),解决分类问题
# ===============================================================================================
Y = [[int(x1+x2 < 1)] for (x1, x2) in X]

with tf.Session() as sess:  # 创建一个回话
    init_op = tf.global_variables_initializer()  # 初始化变量
    sess.run(init_op)  # 使用创建号的回话来得到关心的运算的结果
    print('w1:', sess.run(w1))
    print('w2:', sess.run(w2))
    print("\n")

    STEPS = 5000
    for i in range(STEPS):
        start = (i * batch_size) % dataset_size  # batch_size=8,dataset_size = 128,%表示取模,返回除法的余数
        end = (i * batch_size) % dataset_size + batch_size
        # ===============================================================================================
        # 1. sess.run()参数可以是list
        # 2. feed_dict:给使用placeholder创建出来的tensor赋值
        # ===============================================================================================
        sess.run([train_step, y, y_], feed_dict={x: X[start:end], y_: Y[start:end]})
        if i % 1000 == 0:
            total_cross_entropy = sess.run(cross_entropy, feed_dict={x: X, y_: Y})
            print("After %d training step(s), cross entropy on all Data is %g" % (i, total_cross_entropy))

    print("\n")
    print(sess.run(w1))
    print(sess.run(w2))