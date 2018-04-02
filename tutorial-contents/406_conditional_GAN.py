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
N_IDEAS = 5             # think of this as number of ideas for generating an art work (Generator)
ART_COMPONENTS = 15     # it could be total point G can draw in the canvas
PAINT_POINTS = np.vstack([np.linspace(-1, 1, ART_COMPONENTS) for _ in range(BATCH_SIZE)])    # shape(64行, 15列)

# show our beautiful painting range
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + 1, c='#74BCFF', lw=3, label='upper bound')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + 0, c='#FF9359', lw=3, label='lower bound')
plt.legend(loc='upper right')
plt.show()


def artist_works():     # 来自著名艺术家的绘画(真实目标) (real target)
    a = np.random.uniform(1, 2, size=BATCH_SIZE)[:, np.newaxis]
    paintings = a * np.power(PAINT_POINTS, 2) + (a-1)
    labels = (a - 1) > 0.5  # 艺术家画了两种类别的画
    labels = labels.astype(np.float32)
    return paintings, labels      # paintings.shape(64, 15);labels.shape(64, 1)

art_labels = tf.placeholder(tf.float32, [None, 1])    # G和D的nn的输入多加入了真实作品的art_labels的特征
with tf.variable_scope('Generator'):
    G_in = tf.placeholder(tf.float32, [None, N_IDEAS])          # 输入随机想法(可能来自正态分布),shape(64, 5)
    G_art = tf.concat((G_in, art_labels), axis=1)               # 输入shape(64, 6)
    G_l1 = tf.layers.dense(G_art, 128, tf.nn.relu)
    G_out = tf.layers.dense(G_l1, ART_COMPONENTS)               # 生成器输出shape(64, 15)

with tf.variable_scope('Discriminator'):
    real_in = tf.placeholder(tf.float32, [None, ART_COMPONENTS], name='real_in')   # receive art work from the famous artist + label
    real_art = tf.concat((real_in, art_labels), axis=1)                            # 输入nn的shape(64, 16)
    D_l0 = tf.layers.dense(real_art, 128, tf.nn.relu, name='l')
    prob_artist0 = tf.layers.dense(D_l0, 1, tf.nn.sigmoid, name='out')              # 得到作品由艺术家创作的可能性,输出shpe(64, 1)
    
    # 鉴别Generator的作品
    G_art = tf.concat((G_out, art_labels), 1)                                       
    D_l1 = tf.layers.dense(G_art, 128, tf.nn.relu, name='l', reuse=True)            
    prob_artist1 = tf.layers.dense(D_l1, 1, tf.nn.sigmoid, name='out', reuse=True)  # probability that the art work is made by artist

D_loss = -tf.reduce_mean(tf.log(prob_artist0) + tf.log(1-prob_artist1))
G_loss = tf.reduce_mean(tf.log(1-prob_artist1))

train_D = tf.train.AdamOptimizer(LR_D).minimize(
    D_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator'))
train_G = tf.train.AdamOptimizer(LR_G).minimize(
    G_loss, var_list=tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator'))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

plt.ion()   # something about continuous plotting
for step in range(7000):
    artist_paintings, labels = artist_works()               # real painting from artist
    G_ideas = np.random.randn(BATCH_SIZE, N_IDEAS)
    G_paintings, pa0, Dl = sess.run([G_out, prob_artist0, D_loss, train_D, train_G],    # train and get results
                                    {G_in: G_ideas, real_in: artist_paintings, art_labels: labels})[:3]

    if step % 50 == 0:  # plotting
        plt.cla()
        plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='Generated painting',)
        bound = [0, 0.5] if labels[0, 0] == 0 else [0.5, 1]    # 生成a的边界值,bound
        plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound')
        plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound')
        plt.text(-.5, 1.7, 'Class = %i' % int(labels[0, 0]), fontdict={'size': 15})
        plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.draw(); plt.pause(0.3)
        print('D判断真实作品为真的可能性:{}, D的判别能力的误差:{}'.format(pa0.mean(), Dl))
        # pa0.mean()判断真实作品为真的可能性:从刚开始的高可能性(一眼就认出是真实的作品),到后面接近于50%
        # Dl判别器的误差:从刚开始的比较小,到后面变大趋近于稳定
plt.ioff()

# 让Generateor生成1类的作品
plt.figure(2)
z = np.random.randn(1, N_IDEAS)
label = np.array([[1.]])            # for upper class
G_paintings = sess.run(G_out, {G_in: z, art_labels: label})
plt.plot(PAINT_POINTS[0], G_paintings[0], c='#4AD631', lw=3, label='G painting for upper class',)
plt.plot(PAINT_POINTS[0], 2 * np.power(PAINT_POINTS[0], 2) + bound[1], c='#74BCFF', lw=3, label='upper bound (class 1)')
plt.plot(PAINT_POINTS[0], 1 * np.power(PAINT_POINTS[0], 2) + bound[0], c='#FF9359', lw=3, label='lower bound (class 1)')
plt.ylim((0, 3)); plt.legend(loc='upper right', fontsize=12); plt.show()