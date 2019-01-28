import numpy as np
import random
import matplotlib.pyplot as plt
import tensorflow as tf
import gym

def DQN(env, memory = 10000, batch_size = 32, gamma = 1, learning_rate = 1e-4, decay_rate = 1e-4, C = 1, threshold = 195):

    tf.random.set_random_seed(0)
    np.random.seed(0)
    env.seed(0)
    random.seed(0)

    obs_size = env.observation_space.shape[0]
    act_size = env.action_space.n

    x = tf.placeholder(dtype=tf.float32, shape=[obs_size, None])
    x_h = tf.placeholder(dtype=tf.float32, shape=[obs_size, None])

    W1 = tf.get_variable('W1', [64, obs_size], initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable('b1', [64, 1], initializer=tf.zeros_initializer())
    H = tf.nn.relu(tf.add(tf.matmul(W1, x), b1))
    W2 = tf.get_variable('W2', [act_size, 64], initializer=tf.contrib.layers.xavier_initializer())
    q = tf.matmul(W2, H)

    W1_h = tf.get_variable('W1_h', initializer=W1, trainable=False)
    b1_h = tf.get_variable('b1_h', initializer=b1, trainable=False)
    H_h = tf.nn.relu(tf.add(tf.matmul(W1_h, x_h), b1_h))
    W2_h = tf.get_variable('W2_h', initializer=W2, trainable=False)
    q_h = tf.matmul(W2_h, H_h)

    y = tf.placeholder(dtype=tf.float32, shape=[act_size, None])

    mask = tf.placeholder(dtype=tf.int32, shape=[act_size, None])
    loss = tf.losses.mean_squared_error(y, q, weights = mask)

    assign_op = [tf.assign(W1_h, W1), tf.assign(W2_h, W2), tf.assign(b1_h, b1)]
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    c = 0
    episode_count = 0
    late_count = 0
    R = []
    D = []

    with tf.Session() as sess:

        sess.run(tf.global_variables_initializer())

        while True:

            if episode_count % 100 == 0:

                r = 0
                for j in range(100):

                    s = env.reset()

                    while True:

                        # if j == 0:
                        #     env.render()

                        Q = sess.run(q, feed_dict={x: np.array(s).reshape(obs_size, 1)})
                        a = np.argmax(Q)
                        info = env.step(a)
                        r += info[1]
                        s = info[0]
                        if info[2]:
                            break

                print(r/100)
                R.append(r/100)

                if R[-1] >= threshold:
                    break

            s = env.reset()
            epsilon = 0.5 - min(decay_rate * (episode_count - late_count), 0.49)
            episode_count += 1

            while True:

                tmp = np.random.rand()
                Q = sess.run(q, feed_dict = {x: np.array(s).reshape(obs_size, 1)})
                if tmp > epsilon:
                    a = np.argmax(Q)
                else:
                    a = np.random.randint(act_size)

                info = env.step(a)
                transition = [s, a, info[1], info[0], info[2]]
                s = info[0]
                D.append(transition)
                if len(D) > memory:
                    del D[0]

                if len(D) >= batch_size:

                    sample = random.sample(range(len(D)), batch_size)
                    X = np.zeros((obs_size, batch_size))
                    X_ = np.zeros((obs_size, batch_size))
                    Y = np.zeros((act_size, batch_size))
                    m = np.zeros((act_size, batch_size))
                    for j in range(batch_size):
                        tmp = D[sample[j]]
                        X[:, j] = np.array(tmp[0])
                        X_[:, j] = np.array(tmp[3])
                        Y[:, j] = tmp[2]
                        m[tmp[1], j] = 1
                    Q_h = sess.run(q_h, feed_dict={x_h: X_})

                    for j in range(batch_size):
                        if not D[sample[j]][4]:
                            Y[:, j] += gamma * np.max(Q_h[:, j])
                    sess.run(train_op, feed_dict={x: X, y: Y, mask: m})

                    c += 1
                    if c == C:
                        sess.run(assign_op)
                        c = 0

                if info[2]:
                    break

            if len(D) < batch_size:
                late_count += 1

    return R

env_C = gym.make('CartPole-v0')
R = DQN(env_C)
# env_M = gym.make('MountainCar-v0')
# R = DQN(env_M, learning_rate = 1e-3, decay_rate = 1e-4, C = 10, threshold = -116)

plt.plot(np.arange(len(R)) * 100, R)
plt.xlabel('# of episodes')
plt.ylabel('average reward over 100 episodes')
plt.show()