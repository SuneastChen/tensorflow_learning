# _*_ coding:utf-8 _*_
# !/usr/bin/python

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print('----------------------生成线性模型---------------------')
'''
# 1.创建数据
x_data = np.random.rand(100).astype(np.float32)
# print(x_data)  # 生成一维数组
y_data = x_data*0.1 + 0.3


# 2.创建tensorflow结构
Weights = tf.Variable(tf.random_uniform([1], -1.0, 1.0))  # 生成一个值的一维的权重变量,在-1.0到1.0之间
biases = tf.Variable(tf.zeros([1]))  # 定义初始偏置,有一个0的一维数组

# 预测值, 误差值, 优化器 三大主体
y = Weights*x_data + biases
loss = tf.reduce_mean(tf.square(y-y_data))  # 预测值与结果的差异
optimizer = tf.train.GradientDescentOptimizer(0.5)  # 建立一个优化器,0.5是学习效率,太大波动大,太小效率慢
train = optimizer.minimize(loss)  # 目标使误差最小

init = tf.global_variables_initializer()  # 初始化结构
# -------------tensorflow结构创建完成

sess = tf.Session()  # 创建一个指针
sess.run(init)  # 指针激活结构,重要的一步

for step in range(201):
    sess.run(train)
    if step % 20 == 0:
        print(step, sess.run(Weights), sess.run(biases))
'''



print('----------------------矩阵相乘与Session')
A = tf.constant([[3, 2, 1]])
B = tf.constant([[4],
                 [5],
                 [6]])
product = tf.matmul(A, B)

# 方法一:
sess = tf.Session()
result = sess.run(product)
print(result)
sess.close()

# 方法二:
with tf.Session() as sess:
    result2 = sess.run(product)
    print(result2)

print('----------------------tf.Variable变量')
state = tf.Variable(0, name='counter')  # 定义变量=0
# print(state.name)  # ---> counter:0

one = tf.constant(1)  # 定义常量=1
# print(one.name)  # ---> Const_2:0

add_value = tf.add(state, one)  # new_value是加法命令
update = tf.assign(state, add_value)  # update也是一个赋值命令


init = tf.global_variables_initializer()  # 初始化结构
with tf.Session() as sess:
    sess.run(init)  # 激活结构

    for _ in range(5):
        sess.run(update)  # 执行方法
        print(sess.run(state))

print('----------------------placeholder形参!')
input1 = tf.placeholder(tf.float32)  # (tf.float32, [3,2]) 也可以加入几行几列参数,tf.float32必须要加
input2 = tf.placeholder(tf.float32)  # tf.placeholder()在sess.run()时传入实参
output = tf.add(input1, input2)
with tf.Session() as sess:
    print(sess.run(output, feed_dict={input1: [2.0], input2: [7.0]}))  # 用feed_dict字典形式传入实参



print('----------------------多层神经网络(1,10,1),非线性---------------------')
'''
def add_layer(inputs, in_size, out_size, activation_function=None):
    # inputs输入值,in_size输入神经元的个数,out_size本层神经元的个数,activation_function激活函数
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))  # 权重二维数组,行数为输入神经元个数,列数为输出神经元个数
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)  # biases一般不为0
    Wx_plus_b = tf.matmul(inputs, Weights) + biases  # y = kx + b
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)   # 使y的值在+/-1之间或0--1之间
    return outputs

# 原始数据
x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.5, x_data.shape)  # mean=0,方差=0.05,x_data.shape格式(300,1)
y_data = np.square(x_data) + noise

x_data_ = tf.placeholder(tf.float32, [None, 1])   # 输入的数据 变量(形参),None表示任何行数都可以,即记录的数量;1表示特征的输入数量
y_data_ = tf.placeholder(tf.float32, [None, 1])   # 输出的数据 变量(形参),用于run()时,传递实参;None表示记录的数量,1表示输出的种类数量



# 有几个特征值,输入层就有几个神经元;输出层也一样;如下为主体结构
layer1 = add_layer(x_data_, 1, 10, activation_function=tf.tanh)  # 定义隐藏层,可加入不同的激励函数类型
prediction = add_layer(layer1, 10, 1, activation_function=None)  # 定义输出层(预测,预言),无激励函数

loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data_ - prediction), reduction_indices=[1]))
       # tf.reduce_mean()求平均,tf.reduce_sum()求和,tf.square()求平方

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)  # 训练的目标:创建一个优化器使loss变小


init = tf.global_variables_initializer()  # 初始化结构
sess = tf.Session()
sess.run(init)
'''


# 1.直接输出误差结果
'''
loss_dict = {}
for i in range(1000):
    sess.run(train_step, feed_dict={x_data_: x_data, y_data_: y_data})
    if i % 25 == 0:
        res_loss = sess.run(loss, feed_dict={x_data_: x_data, y_data_: y_data})
        loss_dict[i] = res_loss

# 误差可视化图
loss_s = pd.Series(loss_dict)
loss_s.plot()
plt.show()

'''

# 2.拟合过程的可视化
'''
fig = plt.figure()  # 建一个轮廓窗口
ax = fig.add_subplot(1, 1, 1)  # 增加子图(1,1),取第1个
ax.scatter(x_data, y_data)
plt.ion()  # 展现图表时,程序不暂停
plt.show()

for i in range(1000):
    sess.run(train_step, feed_dict={x_data_: x_data, y_data_: y_data})
    if i % 25 == 0:
        try:
            ax.lines.remove(lines[0])  # 画布上的线删除,第一条线
        except Exception:
            pass
        prediction_value = sess.run(prediction, feed_dict={x_data_: x_data})
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)  # 画布画曲线,'r-'为红色,lw=5为线宽
        plt.pause(0.5)  # 画布停留0.5秒
'''

print('----------------------用上例,Tensorboard可视化---------------------')
'''
def add_layer(inputs, in_size, out_size, activation_function=None):
    with tf.name_scope('weights'):  # 流程图中显示weights
        Weights = tf.Variable(tf.random_normal([in_size, out_size]), name='W')
    with tf.name_scope('biases'):
        biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, name='b')
    with tf.name_scope('Wx_plus_b'):
        Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

x_data = np.linspace(-1, 1, 300)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape)
y_data = np.square(x_data) + noise

with tf.name_scope('inputs_1'):  # 输入的大框框
    x_data_ = tf.placeholder(tf.float32, [None, 1], name='x_input')  # 加入name参数,用于显示
    y_data_ = tf.placeholder(tf.float32, [None, 1], name='y_input')

with tf.name_scope('layer_10'):  # 隐藏层的大框框
    layer1 = add_layer(x_data_, 1, 10, activation_function=tf.tanh)
with tf.name_scope('outputs_1'):  # 输出的大框框
    prediction = add_layer(layer1, 10, 1, activation_function=None)

with tf.name_scope('loss'):   # 计算误差loss的大框框
    loss = tf.reduce_mean(tf.reduce_sum(tf.square(y_data_ - prediction), reduction_indices=[1]))
with tf.name_scope('train'):  # 最终训练的大框框
    train_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)


sess = tf.Session()
writer = tf.summary.FileWriter('logs/', sess.graph)  # 整个图像写入文件

init = tf.global_variables_initializer()
sess.run(init)

# 在cmd 中进入目录(不要有中文目录),先输入tensorboard --inspect --logdir logs
# 再输入 tensorboard --logdir logs,再用浏览器打开'localhost:6006'网址查看
'''


print('----------------------神经网络进行,分类问题---------------------')
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 如果没有数据包,会从网上下载

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

def compute_accuracy(v_xs, v_ys):  # 计算准确度
    y_pre = sess.run(prediction, feed_dict={x_data_: v_xs})  # 得到概率最大的预测值
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # 预测结果是否相等
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求平均正确的概率
    result = sess.run(accuracy, feed_dict={x_data_: v_xs, y_data_: v_ys})  # 执行函数
    return result

x_data_ = tf.placeholder(tf.float32, [None, 784])  # 一张图片上28*28=784个点
y_data_ = tf.placeholder(tf.float32, [None, 10])   # 每张图片对应一行的10个点的结果


# # 预测值, 误差值, 优化器 三大主体
prediction = add_layer(x_data_, 784, 10, activation_function=tf.nn.softmax)  # 一张图片对应输入点数784,输出点数10
# tf.nn.softmax 配合cross_entropy(交叉熵)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data_ * tf.log(prediction), reduction_indices=[1]))  # 就相当于loss
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)  # 每100张图片进行分割train,test
    sess.run(train_step, feed_dict={x_data_: batch_x, y_data_: batch_y})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))  # 传入已分割的测试图片,输出预测的正确率
'''



print('----------------------图片识别神经网络的进化,卷积神经网络CNN---------------------')

'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)


def compute_accuracy(v_xs, v_ys):  # 计算准确度
    y_pre = sess.run(prediction, feed_dict={x_data_: v_xs})  # 得到概率最大的预测值
    correct_prediction = tf.equal(tf.argmax(y_pre, 1), tf.argmax(v_ys, 1))  # 预测结果是否相等
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # 求正确数量平均
    result = sess.run(accuracy, feed_dict={x_data_: v_xs, y_data_: v_ys, keep_prob: 0.9})  # 即得到平均正确的概率
    return result

def weight_variable(shape):  # 定义权重变量
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):  # 定义偏置变量
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):  # 定义2d的扫描器
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    # stride的格式[1, x_movement, y_movement, 1] (前后必须要等于1), paddint='SAME'(边上也会扫描)/'VALID'

def max_pool_2x2(x):  # 定义磁化,不产生数据丢失
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')


x_data_ = tf.placeholder(tf.float32, [None, 784])  # 一张图片上28*28=784个点
y_data_ = tf.placeholder(tf.float32, [None, 10])   # 每张图片对应一行的10个点的结果
keep_prob = tf.placeholder(tf.float32)

x_image = tf.reshape(x_data_, [-1, 28, 28, 1])  # -1接受所有图片个数,28(长)*28(宽), 1代表通道,黑白图片为1,彩色为3(RGB)
# print(x_image.shape)  # [n_samples, 28, 28, 1]

# 压缩层1
W_conv1 = weight_variable([5, 5, 1, 32])  # patch(扫描的小方块大小)5*5 ,in size(图片的输入厚度)=1, out size(图片输出厚度)=32
b_conv1 = bias_variable([32])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 输出28(长)*28(宽)*32(厚)
h_pool1 = max_pool_2x2(h_conv1)   # 输出14*14*32,因为定义磁化的步长为2

# 压缩层2
W_conv2 = weight_variable([5, 5, 32, 64])  # patch(扫描的小方块大小)5*5 ,输入厚度=32, 输出厚度=64
b_conv2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 根据上个压缩层1的结果,输出14(长)*14(宽)*64(厚)
h_pool2 = max_pool_2x2(h_conv2)   # 输出7*7*64,因为定义磁化的步长为2


# 定义神经网络的隐藏层1
W_fc1 = weight_variable([7*7*64, 124])  # 权重的矩阵的行数 = 输入层的特征数量,即输入层神经元个数
                                          # 权重矩阵的列数 = 本层的神经元个数
b_fcl = bias_variable([124])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])  # 将立体转平面[n_samples,7,7,64] ---> [n_samples,7*7*64]
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fcl)   # 输出结果,让激励函数处理
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)


# 隐藏层2
W_fc2 = weight_variable([124, 10])  # 权重的矩阵的行数 = 输入层的特征数量,即输入层神经元个数
                                          # 权重矩阵的列数 = 本层的神经元个数
b_fc2 = bias_variable([10])
prediction = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)   # 输出结果,让softmax函数处理
# tf.nn.softmax 配合cross_entropy(交叉熵)
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_data_ * tf.log(prediction), reduction_indices=[1]))  # 就相当于loss
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)  # AdamOptimizer优化器


sess = tf.Session()
sess.run(tf.global_variables_initializer())


for i in range(1000):
    batch_x, batch_y = mnist.train.next_batch(100)  # 每100张图片进行分割train,test
    sess.run(train_step, feed_dict={x_data_: batch_x, y_data_: batch_y, keep_prob: 0.9})
    if i % 50 == 0:
        print(compute_accuracy(mnist.test.images, mnist.test.labels))  # 传入已分割的测试图片,输出预测的正确率
'''



print('----------------------神经网络的保存与读取---------------------')
# 只能保存变量,不能保存整个神经网络,需要导入变量,重新定义框架,再训练学习
# 一个py文件中保存变量结构功能与读取不可同时存在,会报错
'''
# 保存时,定义变量必须要加上dtype参数,且用float32,还要有name参数,后续才能根据name导入
W = tf.Variable([[3, 2, 1], [4, 5, 6]], dtype=tf.float32, name='weights')
b = tf.Variable([[7, 8, 9]], dtype=tf.float32, name='biases')

init = tf.global_variables_initializer()

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)  # 先激活init结构
    save_path = saver.save(sess, 'my_net/save_net.ckpt')  # 再保存
    print('保存路径:', save_path)
'''


'''
# 导入时重新定义变量,shape与dtype参数,需保持一致; 不需要init
W1 = tf.Variable(np.arange(6).reshape((2, 3)), dtype=tf.float32, name='weights')
b1 = tf.Variable(np.arange(3).reshape((1, 3)), dtype=tf.float32, name='biases')

saver = tf.train.Saver()
with tf.Session() as sess:
    saver.restore(sess, 'my_net/save_net.ckpt')
    print('weights变量:', sess.run(W1))
    print('biases变量:', sess.run(b1))

'''



print('----------------------循环神经网络(RNN分类例子)---------------------')
'''
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)  # 原始数据

lr = 0.001  # 学习率
training_iters = 90000  # 训练次数,即图片总数量
batch_size = 128   # 每一批输入128张图片

n_inputs = 28  # 图片的shape=28*28,输入一行的特征数据,即28列
n_steps = 28  # 时间轴的单位时间步长为28行
n_hidden_unis = 128  # cell层的神经元个数
n_classes = 10   # 输出结果的特征数量

x_ = tf.placeholder(tf.float32, [None, n_steps, n_inputs])  # 输入数据的形参,一个单位时间=(28行),(128张,28行,28个列特征输入)
y_ = tf.placeholder(tf.float32, [None, n_classes])  # 一个单位时间(28行)产生一个输出数据,(128张,10个输出类型)

weights = {
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_unis])),  # 即输入层的W,入28,出128
    'out': tf.Variable(tf.random_normal([n_hidden_unis, n_classes]))  # 即输出层的W,入128,出10
}
biases = {
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_unis])),
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes]))
}

states = []
def RNN(X,weights,biases):
    # --------定义输入层,为cell提供输入
    # X.shape为转主线转换后的(128个小批量图,28行即steps,28列即inputs)
    X = tf.reshape(X, [-1, n_inputs])  # --->>转换成(128个图*28行,28列)
    X_in = tf.matmul(X, weights['in']) + biases['in']  # X_in.shape=(128个图*28行,128列)
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_unis])  # --->>转换成(128个图, 28行, 128列)

    # --------定义cell
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(n_hidden_unis, forget_bias=1.0, state_is_tuple=True)  # 定义cell层处理对象
    global states
    if states == []:
        _init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)  # _init_state,是个元组,(c_state主线的经验,m_state本次经验)
    else:
        _init_state = states
    outputs, states = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=_init_state, time_major=False)  # time_major时间维度是否在第三维上(128张),不是,在28行上
    # outputs为cell层输出的数据,states为[以前累计经验,和本次的升级经验]的列表
    # print(outputs.shape)  # --->> (128, 28, 128)

    # --------定义输出前的层,接收cell的产生的states输入
    # 方法1:
    # 只有当单位时间的所有输入列,即行与列产生一个结果时,才可以用states[1]代替outputs的数据(128张,28行,128列)
    # results = tf.matmul(states[1], weights['out']) + biases['out']   # state[1]==output[-1]

    # 方法2:
    # 28行的数据只产生一个结果,所以要在每张图片的第28行生成results
    outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # 原为[0,1,2],故三维与二维互换,得到shape=(28,128,128),然后解开得到(128*128)的矩阵列表,取最后一个就是真正的结果
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    # print(results.shape)  # ---->> (128,10)
    
    return results



pred = RNN(x_, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y_))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)

correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y_, 1))  # 预测结果是否相等
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))   # 得到正确率

init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < training_iters:  # batch_size是一批数量 training_iters是要学习图片总数量
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)   # 生成小批量128张图片数据,及结果数据
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])  # 将二维的图片数据,转为三维的,保持列数不变
        sess.run([train_op], feed_dict={x_: batch_xs, y_: batch_ys})



        if step % 20 == 0:   # 每执行20个小批量打印一下准确度
            print(sess.run(accuracy, feed_dict={x_: batch_xs, y_: batch_ys}))
        step += 1

'''


print('----------------------循环神经网络(RNN回归问题)---------------------')

'''
BATCH_START = 0
TIME_STEPS = 20
BATCH_SIZE = 50
INPUT_SIZE = 1
OUTPUT_SIZE = 1
CELL_SIZE = 10
LR = 0.006
BATCH_START_TEST = 0


def get_batch():   # 生成一个批量的原始数据
    global BATCH_START, TIME_STEPS
    xs = np.arange(BATCH_START, BATCH_START + TIME_STEPS*BATCH_SIZE).reshape((BATCH_SIZE, TIME_STEPS))/10
    # xs是时间线,每隔20个为时间单位,每个小批量产生一个时间单位(不管批量的大小),故小批量取值范围分别是0--1000,20--1020,40--1040......

    # seq = xs   # 当真实结果有一定规律时,不管输入的方程是什么,误差会缩小在一定的范围,拟合到大方向的趋势
    # seq = xs**3 + xs**2 + xs
    seq = np.sin(xs)

    # res = np.random.random((BATCH_SIZE, TIME_STEPS))-0.4   # 当真实结果本身无规律时,随机的,学习无效
    # res = res.cumsum().reshape((BATCH_SIZE, TIME_STEPS))
    res = np.cos(xs) + BATCH_START*0.001

    BATCH_START += TIME_STEPS
    return [seq[:, :, np.newaxis], res[:, :, np.newaxis], xs]

class LSTMRNN(object):
    def __init__(self, n_steps, input_size, output_size, cell_size, batch_size):
        self.n_steps = n_steps  # 时间序列的间隔,20
        self.input_size = input_size  # 初始数据特征个数,1个
        self.output_size = output_size  # 最终输出神经元个数,1个
        self.cell_size = cell_size   # cell个数,即中间的处理层神经元个数,10个
        self.batch_size = batch_size  # 小批量学习数量,50
        with tf.name_scope('inputs'):
            self.xs = tf.placeholder(tf.float32, [None, n_steps, input_size], name='xs')  # 输入数据,(50个单位时间的输入,时间间隔20,1个特征输入)
            self.ys = tf.placeholder(tf.float32, [None, n_steps, output_size], name='ys')  # 输出数据,(50个单位时间的产生的输出,时间间隔20,1个特征输出)
        with tf.variable_scope('in_hidden'):
            self.add_input_layer()  # 初始化就执行的函数
        with tf.variable_scope('LSTM_cell'):
            self.add_cell()
        with tf.variable_scope('out_hidden'):
            self.add_output_layer()
        with tf.name_scope('cost'):
            self.compute_cost()
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(LR).minimize(self.cost)

    def add_input_layer(self):  # 处理输入层
        l_in_x = tf.reshape(self.xs, [-1, self.input_size])  # 将输入的三维数据(50, 20, 1),转为二维数据(50*20,1)
        Ws_in = self._weight_variable([self.input_size, self.cell_size])
        bs_in = self._bias_variable([self.cell_size,])
        with tf.name_scope('Wx_plus_b'):
            l_in_y = tf.matmul(l_in_x, Ws_in) + bs_in   # 得到二维的数据结果
        # 将得到结果再转为三维的数据,数据转换始终保持列数不变
        self.l_in_y = tf.reshape(l_in_y, [-1, self.n_steps, self.cell_size], name='2to3D')



    def add_cell(self):  # 核心处理层
        lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(self.cell_size, forget_bias=1.0, state_is_tuple=True)
                             # 初始不忘记经验,state_is_tuple=True推荐使用,产生(以前累计经验,本次累计经验)的元组
        with tf.name_scope('initial_state'):
            self.cell_init_state = lstm_cell.zero_state(self.batch_size, dtype=tf.float32)

        # outputs为cell层的所有输出数据的list,states为[以前累计经验,和本次的升级经验]的列表,
        # states就是cell层的输出数据,输出前层的输入数据
        self.cell_outputs, self.cell_final_state = tf.nn.dynamic_rnn(
            lstm_cell, self.l_in_y, initial_state=self.cell_init_state, time_major=False)
            # (执行哪个cell, 输入的数据, 带的经验值, 时间线是否在第三维)
            # ---> 产生了仍是三维的outputs,往上给输出层,得到了预测,对比真实结果,产生了最新的经验数据states
        # 第一个cell用初始的self.cell_init_state,执行产生了新的self.cell_final_state,主线程用变量state接收
        # 下一个cell开始执行时,主线程给了最新的state,给self.cell_init_state,执行....
        # 时间线state的传递以元组形式


    def add_output_layer(self):  # 输出前处理层
        l_out_x = tf.reshape(self.cell_outputs, [-1, self.cell_size], name='3to2D')  # 转换成2D数据,才能进行矩阵运算输出结果
        Ws_out = self._weight_variable([self.cell_size, self.output_size])
        bs_out = self._bias_variable([self.output_size,])

        with tf.name_scope('Wx_plus_b'):
            self.pred = tf.matmul(l_out_x, Ws_out) + bs_out



    def compute_cost(self):
        losses = tf.contrib.legacy_seq2seq.sequence_loss_by_example(
            [tf.reshape(self.pred, [-1], name='reshape_pred')],
            [tf.reshape(self.ys, [-1], name='reshape_target')],
            [tf.ones([self.batch_size * self.n_steps], dtype=tf.float32)],
            average_across_timesteps=True,
            softmax_loss_function=self.msr_error,
            name='losses'
        )   # 得到每个单位时间的loss,类型为一个数字
        with tf.name_scope('average_cost'):
            self.cost = tf.div(tf.reduce_sum(losses, name='losses_sum'),
                               tf.cast(self.batch_size, tf.float32),
                               name='average_cost')  # 将loss加总,求平均,得到小批量的一个数字
            tf.summary.scalar('cost', self.cost)

    def msr_error(self, logits, labels):
        return tf.square(tf.subtract(logits, labels))

    def _weight_variable(self, shape, name='weights'):
        initializer = tf.random_normal_initializer(mean=0., stddev=1.)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

    def _bias_variable(self, shape, name='biases'):
        initializer = tf.constant_initializer(0.1)
        return tf.get_variable(name=name, shape=shape, initializer=initializer)

if __name__ == '__main__':
    model = LSTMRNN(TIME_STEPS, INPUT_SIZE, OUTPUT_SIZE, CELL_SIZE, BATCH_SIZE)  # 创建一个模型对象
    sess = tf.Session()

    merged = tf.summary.merge_all()
    writer = tf.summary.FileWriter('RNN_sin_logs', sess.graph)   # 保存后,可以浏览器查看
    sess.run(tf.global_variables_initializer())


    plt.ion()
    plt.show()

    for i in range(500):
        seq, res, xs = get_batch()  # 生成原始数据,req为输入数据,res为结果数据,结构为(50,20,1),xs是时间线
        if i == 0:
            feed_dict = {model.xs: seq, model.ys: res}  # 用初始的state
        else:
            feed_dict = {model.xs: seq, model.ys: res, model.cell_init_state: state}  # 后续步使用最新的state作为经验
        _, cost, state, pred = sess.run([model.train_op, model.cost, model.cell_final_state, model.pred],
                                        feed_dict=feed_dict)  # 执行各种方法
        # print(xs.shape)  # (50, 20)
        # print(res.shape)   # (50, 20, 1)
        # print(pred.shape)   # (1000, 1)


        # 动态可视化, $ numpy的切片,由外到内一层层切片,都从小批量的数据中取20个数据,是单位时间内的数据
        plt.plot(xs[0, :], res[0, :].flatten(), 'r',   # 绘制单位时间内的期望线
                 xs[0, :], pred.flatten()[:TIME_STEPS], 'b--')  # 绘制单位时间内的预测线
        plt.ylim(-1.2, 1.2)
        plt.draw()
        plt.pause(0.3)

        if i % 15 == 0:
            print('cost:', round(cost, 4))  # 循环15次打印第15次的coat,即执行15次小批量时
            result = sess.run(merged, feed_dict)
            writer.add_summary(result, i)

    plt.pause(-1)
'''



print('----------------------自编码,非监督学习的改造---------------------')
# 先提取特征,降低数据量,再导入神经网络重复学习
print('-------1.预测的图片与真实的图片对比')
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)

learning_rate = 0.01
training_epochs = 5   # 重复学习5遍
batch_size = 256
display_step = 1
examples_to_show = 10

n_input = 784  # image shape:28*28

X = tf.placeholder('float', [None, n_input])

n_hidden_1 = 256   # 第一层神经元数256个
n_hidden_2 = 128   # 第二层神经元数128个

# 784 ---> 256 ---> 128
weights = {'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
            'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),

           'decoder_h1': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
            'decoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
           }
biases = {'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),

           'decoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
            'decoder_b2': tf.Variable(tf.random_normal([n_input])),
           }

def encoder(x):  # 经过两层压缩
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    return layer_2

def decoder(x):  # 两层解压缩,激励函数保持一致
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    return layer_2


encoder_op = encoder(X)  # 传入X进行压缩得到encoder_op
decoder_op = decoder(encoder_op)  # 再将encoder_op解压缩


y_pred = decoder_op  # 预测结果
y_true = X  # 实际结果

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)  #有多少的小批量
    for epoch in range(training_epochs):  # 5遍重复训练
        for i in range(total_batch):  # 每个批量进行训练
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)   # x在0--1之间
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print('Epoch:', '%d' % (epoch + 1),
                  'cost=', '{:.9f}'.format(c))

    print('Optimization Finished!')

    encode_decode = sess.run(y_pred, feed_dict={X: mnist.test.images[:examples_to_show]})  # 预测的图片

    f, a = plt.subplots(2, 10, figsize=(10, 2))  # a相当于表格
    for i in range(examples_to_show):
        a[0][i].imshow(np.reshape(mnist.test.images[i], (28, 28)))  # 在a的第1行显示真实的图片
        a[1][i].imshow(np.reshape(encode_decode[i], (28, 28)))  # 在a的第2行显示预测的图片
    plt.show()

'''


print('-------2.坐标轴上显示分类点')
'''
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=False)

learning_rate = 0.001  # 更改学习率
training_epochs = 20   # 重复学习20遍
batch_size = 256
display_step = 1

n_input = 784  # image shape:28*28

X = tf.placeholder('float', [None, n_input])

n_hidden_1 = 128
n_hidden_2 = 64
n_hidden_3 = 10
n_hidden_4 = 2   # 用到四层神经网络

# 784 ---> 256 ---> 128
weights = {'encoder_h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
           'encoder_h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
           'encoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_3])),
           'encoder_h4': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_4])),

           'decoder_h1': tf.Variable(tf.random_normal([n_hidden_4, n_hidden_3])),
           'decoder_h2': tf.Variable(tf.random_normal([n_hidden_3, n_hidden_2])),
           'decoder_h3': tf.Variable(tf.random_normal([n_hidden_2, n_hidden_1])),
           'decoder_h4': tf.Variable(tf.random_normal([n_hidden_1, n_input])),
           }
biases = {'encoder_b1': tf.Variable(tf.random_normal([n_hidden_1])),
          'encoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
          'encoder_b3': tf.Variable(tf.random_normal([n_hidden_3])),
          'encoder_b4': tf.Variable(tf.random_normal([n_hidden_4])),

          'decoder_b1': tf.Variable(tf.random_normal([n_hidden_3])),
          'decoder_b2': tf.Variable(tf.random_normal([n_hidden_2])),
          'decoder_b3': tf.Variable(tf.random_normal([n_hidden_1])),
          'decoder_b4': tf.Variable(tf.random_normal([n_input])),
           }

def encoder(x):  # 经过四层压缩
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['encoder_h1']),
                                   biases['encoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['encoder_h2']),
                                   biases['encoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['encoder_h3']),
                                   biases['encoder_b3']))
    layer_4 = tf.add(tf.matmul(layer_3, weights['encoder_h4']),  # 不用再经过激励函数
                                   biases['encoder_b4'])
    return layer_4

def decoder(x):  # 四层解压缩,激励函数保持一致
    layer_1 = tf.nn.sigmoid(tf.add(tf.matmul(x, weights['decoder_h1']),
                                   biases['decoder_b1']))
    layer_2 = tf.nn.sigmoid(tf.add(tf.matmul(layer_1, weights['decoder_h2']),
                                   biases['decoder_b2']))
    layer_3 = tf.nn.sigmoid(tf.add(tf.matmul(layer_2, weights['decoder_h3']),
                                   biases['decoder_b3']))
    layer_4 = tf.nn.sigmoid(tf.add(tf.matmul(layer_3, weights['decoder_h4']),
                                   biases['decoder_b4']))
    return layer_4


encoder_op = encoder(X)  # 传入X进行压缩得到encoder_op
decoder_op = decoder(encoder_op)  # 再将encoder_op解压缩


y_pred = decoder_op  # 预测结果
y_true = X  # 实际结果

cost = tf.reduce_mean(tf.pow(y_true - y_pred, 2))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    total_batch = int(mnist.train.num_examples/batch_size)  #有多少的小批量
    for epoch in range(training_epochs):  # 5遍重复训练
        for i in range(total_batch):  # 每个批量进和训练
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)   # x在0--1之间
            c, _ = sess.run([cost, optimizer], feed_dict={X: batch_xs})
        if epoch % display_step == 0:
            print('Epoch:', '%d' % (epoch + 1),
                  'cost=', '{:.9f}'.format(c))

    print('Optimization Finished!')

    # 显示压缩之后的数据
    encoder_result = sess.run(encoder_op, feed_dict={X: mnist.test.images})
    plt.scatter(encoder_result[:, 0], encoder_result[:, 1], c=mnist.test.labels)
    plt.show()

'''

print('----------------------变量Variable的操作---------------------')

# 用tf.name_scope()时
with tf.name_scope('a_name_scope'):
    initializer = tf.constant_initializer(value=1)   # 定义常量,作为tf.get_variable()的参数
    # initializer = tf.random_normal_initializer(mean=0., stddev=1.)
    var1 = tf.get_variable(name='varl', shape=[1], dtype=tf.float32, initializer=initializer)   # name_scope不显示
    var11 = tf.get_variable(name='var1', shape=[2], dtype=tf.float32, initializer=initializer)  # 重新赋值时,共同一个变量结构

    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)  # name_scope正常显示
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)  # 重新赋值有版本号,重新创建了一份



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name)   # varl:0
    print(sess.run(var1))  # [ 1.]
    print(var11.name)   # var1:0
    print(sess.run(var11))  # [ 1.  1.]


    print(var2.name)  # a_name_scope/var2:0
    print(sess.run(var2))  # [ 2.]
    print(var21.name)  # a_name_scope/var2_1:0    当变量再次赋值时,多了个 _1 的版本
    print(sess.run(var21))  # [ 2.0999999]


print('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
# 用tf.variable_scope()时
with tf.variable_scope('a_variable_scope'):
    initializer = tf.constant_initializer(value=1)   # 定义常量,作为tf.get_variable()的参数
    # initializer = tf.random_normal_initializer(mean=0., stddev=1.)
    var1 = tf.get_variable(name='varl', shape=[1], dtype=tf.float32, initializer=initializer)   # variable_scope可以正常显示
    var11 = tf.get_variable(name='var1', shape=[2], dtype=tf.float32, initializer=initializer)  # 重新赋值时,共同一个变量结构,同上

    var2 = tf.Variable(name='var2', initial_value=[2], dtype=tf.float32)  # variable_scope正常显示,同上
    var21 = tf.Variable(name='var2', initial_value=[2.1], dtype=tf.float32)   # 重新赋值时,有版本号,同上



with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print(var1.name)   # a_variable_scope/varl:0
    print(sess.run(var1))  # [ 1.]
    print(var11.name)  # a_variable_scope/var1:0
    print(sess.run(var11))  # [ 1.  1.]



    print(var2.name)  # a_name_scope/var2:0
    print(sess.run(var2))  # [ 2.]
    print(var21.name)  # a_name_scope/var2_1:0
    print(sess.run(var21))  # [ 2.0999999]





print('----------------------参数调优,梯度下降可视化---------------------')
from mpl_toolkits.mplot3d import Axes3D
'''
LR = 0.2   # 学习率
REAL_PARAMS = [1.2, 2.5]   # 实际用的参数
INIT_PARAMS = [[5, 4],
               [5, 1],
               [2, 4.5]][2]   # 训练的原始参数,取第几个

x = np.linspace(-1, 1, 200, dtype=np.float32)   # 原始数据x

# Test (1): Visualize a simple linear function with two parameters,
# 用线性函数测试时,学习率调大时,每一步的跨度很大,不能很好找到最低点

# y_fun = lambda a, b: a * x + b
# tf_y_fun = lambda a, b: a * x + b


# Test (2): 非线性函数找最优参数

# y_fun = lambda a, b: a * x**3 + b * x**2
# tf_y_fun = lambda a, b: a * x**3 + b * x**2


# Test (3): 复杂函数的找最优参数
# 初始的训练参数很重要,不然会找到局部的最优参数

y_fun = lambda a, b: np.sin(b*np.cos(a*x))
tf_y_fun = lambda a, b: tf.sin(b*tf.cos(a*x))    # 前提必须要有一个经验公式,但不知道最优参数

noise = np.random.randn(200)/10
y = y_fun(*REAL_PARAMS) + noise         # 原始数据的y

# tensorflow graph
a, b = [tf.Variable(initial_value=p, dtype=tf.float32) for p in INIT_PARAMS]   # 生成a,b原始参数列表
pred = tf_y_fun(a, b)
mse = tf.reduce_mean(tf.square(y-pred))
train_op = tf.train.GradientDescentOptimizer(LR).minimize(mse)

a_list, b_list, cost_list = [], [], []
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for t in range(400):
        a_, b_, mse_ = sess.run([a, b, mse])
        a_list.append(a_); b_list.append(b_); cost_list.append(mse_)    # record parameter changes
        result, _ = sess.run([pred, train_op])                          # training


# 可视化代码:
print('a=', a_, 'b=', b_)
plt.figure(1)
plt.scatter(x, y, c='b')    # plot data
plt.plot(x, result, 'r-', lw=2)   # plot line fitting
# 3D cost figure
fig = plt.figure(2); ax = Axes3D(fig)
a3D, b3D = np.meshgrid(np.linspace(-2, 7, 30), np.linspace(-2, 7, 30))  # parameter space
cost3D = np.array([np.mean(np.square(y_fun(a_, b_) - y)) for a_, b_ in zip(a3D.flatten(), b3D.flatten())]).reshape(a3D.shape)
ax.plot_surface(a3D, b3D, cost3D, rstride=1, cstride=1, cmap=plt.get_cmap('rainbow'), alpha=0.5)
ax.scatter(a_list[0], b_list[0], zs=cost_list[0], s=300, c='r')  # initial parameter place
ax.set_xlabel('a'); ax.set_ylabel('b')
ax.plot(a_list, b_list, zs=cost_list, zdir='z', c='r', lw=3)    # plot 3D gradient descent
plt.show()

'''













