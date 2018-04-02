"""
Know more, visit my Python tutorial page: https://morvanzhou.github.io/tutorials/
My Youtube Channel: https://www.youtube.com/user/MorvanZhou
More about Reinforcement learning: https://morvanzhou.github.io/tutorials/machine-learning/reinforcement-learning/

Dependencies:
tensorflow: 1.1.0
matplotlib
numpy
gym: 0.8.1
"""
import tensorflow as tf
import numpy as np
import gym

tf.set_random_seed(1)
np.random.seed(1)

# Hyper Parameters
BATCH_SIZE = 32
LR = 0.01                   # learning rate
EPSILON = 0.9               # 听话度90%,10%的概率不按Q表来
GAMMA = 0.9                 # 对未来奖惩的衰减率
TARGET_REPLACE_ITER = 100   # 每学习100次,更新一下nn的权重
MEMORY_CAPACITY = 2000      # Q表有2000行,即2000个步骤
MEMORY_COUNTER = 0          # 存储记忆的次数
LEARNING_STEP_COUNTER = 0   # 总学习次数
# 存储记忆的次数 > 总学习次数, 因为刚开始在存储记忆,并没有学习
env = gym.make('CartPole-v0')
env = env.unwrapped
N_ACTIONS = env.action_space.n    # 2个行为
N_STATES = env.observation_space.shape[0]  # 4个数字状态
MEMORY = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))     # initialize memory(2000行, 10列)的记忆,每行有(s, [a, r], s_)

# tf placeholders
tf_s = tf.placeholder(tf.float32, [None, N_STATES])  # shape(32, 4)
tf_a = tf.placeholder(tf.int32, [None, ])   # shape(32, )
tf_r = tf.placeholder(tf.float32, [None, ])   # shape(32,)
tf_s_ = tf.placeholder(tf.float32, [None, N_STATES])  # shape(32, 4)

# 定义nn,输入s,输出每个行为的q值,即Q表
with tf.variable_scope('q'):        # evaluation network
    l_eval = tf.layers.dense(tf_s, 10, tf.nn.relu, kernel_initializer=tf.random_normal_initializer(0, 0.1))   # 10个神经元隐藏层
    q = tf.layers.dense(l_eval, N_ACTIONS, kernel_initializer=tf.random_normal_initializer(0, 0.1))   # 输出q.shape为(32, 2)

with tf.variable_scope('q_next'):   # target network, not to train
    l_target = tf.layers.dense(tf_s_, 10, tf.nn.relu, trainable=False)
    q_next = tf.layers.dense(l_target, N_ACTIONS, trainable=False)   # 输出q_next.shape为(32, 2)

q_target = tf_r + GAMMA * tf.reduce_max(q_next, axis=1)        # 根据q_next,计算出q现实,shape(32,)

a_indices = tf.stack([tf.range(tf.shape(tf_a)[0], dtype=tf.int32), tf_a], axis=1)    # 索引矩阵(32行,2列)
q_wrt_a = tf.gather_nd(params=q, indices=a_indices)     # q为(32, 2), 根据索引矩阵取出q估计值,shape(32,)

loss = tf.reduce_mean(tf.squared_difference(q_target, q_wrt_a))   # 损失函数为q现实与q估计的差异方程
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())


def choose_action(s):    # 输入s,输出action(0或1),内部用到nn
    s = s[np.newaxis, :]
    if np.random.uniform() < EPSILON:
        # forward feed the observation and get q value for every actions
        actions_value = sess.run(q, feed_dict={tf_s: s})    # 输入s产生q值,即每个行为的价值
        action = np.argmax(actions_value)
    else:
        action = np.random.randint(0, N_ACTIONS)
    return action


def store_transition(s, a, r, s_):  # 更新每步的记忆
    global MEMORY_COUNTER
    transition = np.hstack((s, [a, r], s_))   # transition为最新的一步记忆
    index = MEMORY_COUNTER % MEMORY_CAPACITY
    MEMORY[index, :] = transition    # 更新相应步骤的记忆
    MEMORY_COUNTER += 1


def learn():    # 进行一次batch_size的学习
    global LEARNING_STEP_COUNTER
    if LEARNING_STEP_COUNTER % TARGET_REPLACE_ITER == 0:    # 每学习100次,更新一下nn的参数
        t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_next')  # 提取神经网络的参数
        e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q')   
        sess.run([tf.assign(t, e) for t, e in zip(t_params, e_params)])  # 更新参数
    LEARNING_STEP_COUNTER += 1

    sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)   # 从2000的范围内随机产生32个整数
    b_memory = MEMORY[sample_index, :]    # 取出部分的记忆行
    b_s = b_memory[:, :N_STATES]
    b_a = b_memory[:, N_STATES].astype(int)
    b_r = b_memory[:, N_STATES+1]
    b_s_ = b_memory[:, -N_STATES:]
    sess.run(train_op, {tf_s: b_s, tf_a: b_a, tf_r: b_r, tf_s_: b_s_})   

print('\nCollecting experience...')
for i_episode in range(400):
    s = env.reset()     # array([-0.00598819,  0.03928614, -0.00193465, -0.02960887])
    ep_r = 0
    while True:
        env.render()    # 渲染环境显示
        a = choose_action(s)   # 在s状态选行为,(一部分随机,一部分用到了nn)

        s_, r, done, info = env.step(a)  # 执行行为
        # (array([-0.01533651, -0.15581019,  0.01380939,  0.26252395]), 1.0, False, {})
        # 修改奖励
        x, x_dot, theta, theta_dot = s_
        r1 = (env.x_threshold - abs(x)) / env.x_threshold - 0.8
        r2 = (env.theta_threshold_radians - abs(theta)) / env.theta_threshold_radians - 0.5
        r = r1 + r2
        # x 是平衡木的水平位移, 所以 r1 是车越偏离中心, 分越少
        # theta 是棒子的偏移角度, 角度越大, 越不垂直. 所以 棒越垂直, r2分越高

        store_transition(s, a, r, s_)   # 每次的行为,都要更新每步的记忆

        ep_r += r
        if MEMORY_COUNTER > MEMORY_CAPACITY:   # 直到存储记忆的次数大于总的步骤数(Q表的行数),才开始学习
            learn()      # 每次学习只随机部分的神经元训练,当每学习到100次,就会更新一下nn的全部参数
            if done:
                print('Ep: ', i_episode,
                      '| Ep_r: ', round(ep_r, 2))

        if done:
            break
        s = s_