"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 64
LR_G = 0.0001           # learning rate for generator
LR_D = 0.0001           # learning rate for discriminator
N_IDEAS = 5             # 生成器的想法的数量,即生成器的输入特征数
ART_COMPONENTS = 15     # 15个点构成一幅画
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])    # shape(64行, 15列)

# show our beautiful painting range
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()


def artist_works():     # 来自著名艺术家的绘画(真实目标) (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]   # 从一个均匀分布[low,high)中随机采样小数,size个; a.shape(64, 1)
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)    # paintings.shape(64, 15),因为a是随机的,所以每行都不一样
    return paintings


with tf.variable_scope('Generator'):   # 生成器神经网络
    G_in = tf.placeholder(tf.float32, [None, N_IDEAS])          # 输入随机想法(可能来自正态分布),shape(64, 5)
    G_l1 = tf.layers.dense(G_in, 128, tf.nn.relu)
    G_out = tf.layers.dense(G_l1, ART_COMPONENTS)               # 通过神经网络一行产生一幅画,shape(64, 15)

with tf.variable_scope('Discriminator'):  # 判别器神经网络
    real_art = tf.placeholder(tf.float32, [None, ART_COMPONENTS], name='real_in')   # 定义tf的参数
    D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name='l')                     # 接受著名艺术家的艺术作品,输入shape(64, 15)
    prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')              # 得到作品由艺术家创作的可能性,输出shape(64, 1)

    # reuse layers for generator
    D_l1 = tf.layers.dense(G_out, 128, tf.nn.relu, name='l', reuse=True)            # 接受像G这样的新手的艺术作品,输入shape(64, 15)
    prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)  # 得到作品由艺术家创作的可能性,输出shape(64, 1)

D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1-prob_artist1))        # 判别器的误差由二分类的损失函数确定
G_loss = tf.reduce_mean(tf.log(1-prob_artist1))                                # 生成器的误差函数

train_D = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   # 程序不会中止,继续画图
for step in range(5000):
    artist_paintings = artist_works()           # 真实的艺术画
    G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)
    G_paintings, pa0, Dl = sess.run([G_out, prob_artist0, D_loss, train_D, train_G],    # train and get results
                                    {G_in: G_ideas, real_art: artist_paintings})[:3]

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting',)
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 2.3, 'D accuracy=%.2f (0.5 for D to converge)' % pa0.mean(), fontdict={'size': 15})
        plt.text(-.5, 2, 'D score= %.2f (-1.38 for G to converge)' % -Dl, fontdict={'size': 15})
        plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.draw(); plt.pause(0.01)

plt.ioff()
plt.show()