import numpy as np
import matplotlib.pyplot as plt
import gym
import mytaxi_2

env_v4 = gym.make('Taxi-v4').unwrapped
env_v5 = gym.make('Taxi-v5').unwrapped

def qlearn(env, gamma = 1, alpha = 1, epsilon = 0.1, runs = 20, episodes = 100):

    np.random.seed(3)
    env.seed(5)
    cum_steps = np.zeros(episodes)

    for k in range(runs):

        Q = np.zeros((env.nS, env.nA))
        step_count = 0

        for j in range(episodes):

            s = env.reset()

            while True:

                greedy = np.random.rand()
                if greedy < epsilon:
                    action = np.random.randint(env.nA)
                else:
                    tmp = np.argwhere(Q[s] == max(Q[s]))
                    action = int(tmp[np.random.randint(len(tmp))].squeeze())

                info = env.step(action)
                step_count += 1
                Q[s][action] += alpha*(info[1] + gamma*max(Q[info[0]]) - Q[s][action])
                s = info[0]

                if info[2]:
                    cum_steps[j] = cum_steps[j] + (step_count - cum_steps[j]) / (k + 1)
                    break

    return Q, cum_steps

def dynaq_a(env, n = 10, gamma = 1, alpha = 1, epsilon = 0.1, episodes = 100):

    np.random.seed(3)
    env.seed(5)
    cum_steps = np.zeros(episodes)
    runs = 20

    for k in range(runs):

        step_count = 0
        model = {}
        Q = np.zeros((env.nS, env.nA))

        for j in range(episodes):

            s = env.reset()

            while True:

                tmp = np.random.rand()
                if tmp > epsilon:
                    a = np.argmax(Q[s])
                else:
                    a = np.random.randint(env.nA)
                info = env.step(a)
                step_count += 1
                r = info[1]
                Q[s, a] = Q[s, a] + alpha*(r + gamma * max(Q[info[0]]) - Q[s, a])
                tmp = env.nA * s + a
                s = info[0]
                if tmp in model:
                    model[tmp].append([r, s])
                else:
                    model[tmp] = [[r, s]]
                if info[2]:
                    cum_steps[j] = cum_steps[j] + (step_count - cum_steps[j]) / (k + 1)
                    break

                for i in range(n):

                    tmp = np.random.randint(len(model))
                    l = 0
                    for sa in model:
                        if l == tmp:
                            sa_p = sa
                            break
                        l += 1
                    s_p = int(sa_p / env.nA)
                    a_p = sa_p % env.nA
                    tmp = np.random.randint(len(model[sa_p]))
                    Q[s_p, a_p] = Q[s_p, a_p] + alpha * (model[sa_p][tmp][0] + gamma * max(Q[model[sa_p][tmp][1]]) - Q[s_p, a_p])

    return Q, cum_steps

def dynaq_b(env_first, env_second, n = 10, gamma = 1, alpha = 1, epsilon = 0.1, episodes = 300, method = 'dynaq'):

    env = env_first
    np.random.seed(3)
    env.seed(5)
    cum_steps = np.zeros(episodes)
    runs = 5
    if method == 'dynaq+':
        K = 0.2
    else:
        K = 0

    for k in range(runs):

        env = env_first
        model = {}
        Q = np.zeros((env.nS, env.nA))
        step_count = 0

        for j in range(episodes):

            T = np.zeros((env.nS, env.nA))
            s = env.reset()

            if j >= 100:
                env = env_second

            while True:

                tmp = np.random.rand()
                if tmp > epsilon:
                    a = np.argmax(Q[s])
                else:
                    a = np.random.randint(env.nA)
                info = env.step(a)
                step_count += 1
                r = info[1]
                Q[s, a] = Q[s, a] + alpha*(r + gamma * max(Q[info[0]]) - Q[s, a])
                tmp = env.nA * s + a
                T[s] = T[s] + 1
                T[s, a] = 0
                s = info[0]
                if tmp in model:
                    model[tmp].append([r, s])
                    if method == 'dynaq_decay' and len(model[tmp]) > 10:
                        del model[tmp][0]
                else:
                    model[tmp] = [[r, s]]
                if info[2]:
                    cum_steps[j] = cum_steps[j] + (step_count - cum_steps[j]) / (k + 1)
                    break

                for i in range(n):

                    tmp = np.random.randint(len(model))
                    l = 0
                    for sa in model:
                        if l == tmp:
                            sa_p = sa
                            break
                        l += 1
                    s_p = int(sa_p / env.nA)
                    a_p = sa_p % env.nA
                    tmp = np.random.randint(len(model[sa_p]))
                    r_p = model[sa_p][tmp][0]
                    if method == 'dynaq+':
                        r_p += K*np.sqrt(T[s_p, a_p])
                    Q[s_p, a_p] = Q[s_p, a_p] + alpha * (r_p + gamma * max(Q[model[sa_p][tmp][1]]) - Q[s_p, a_p])

    return Q, cum_steps

Q_q, cum_steps_q = qlearn(env_v4)
Q_dynaq, cum_steps_dynaq = dynaq_a(env_v4)
plt.plot(cum_steps_q, color = 'r', label = 'Q-learning')
plt.plot(cum_steps_dynaq, color = 'b', label = 'Dyna-Q')
plt.xlabel('episode')
plt.ylabel('cumulative steps')
plt.title('Q-learning and Dyna-Q')
plt.legend()
plt.show()

Q, cum_steps = dynaq_b(env_v4, env_v5, method = 'dynaq')
Q_plus, cum_steps_plus = dynaq_b(env_v4, env_v5, method = 'dynaq+')
Q_decay, cum_steps_decay = dynaq_b(env_v4, env_v5, method = 'dynaq_decay')
plt.plot(cum_steps, color = 'b', label = 'Dyna-Q')
plt.plot(cum_steps_plus, color = 'r', label = 'Dyna-Q+')
plt.plot(cum_steps_decay, color = 'g', label = 'Dyna-Q with model decay')
plt.xlabel('episode')
plt.ylabel('cumulative steps')
plt.title('Dyna-Q and modifications')
plt.legend()
plt.show()