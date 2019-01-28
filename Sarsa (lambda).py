import numpy as np
import matplotlib.pyplot as plt
import gym
from gym.envs.registration import register
from tiles3 import tiles, IHT

register(
    id = 'MountainCar-v1',
    entry_point = 'gym.envs.classic_control:MountainCarEnv',
    max_episode_steps = 5000
)
env = gym.make('MountainCar-v1')

dim = 8
d = 4096
iht = IHT(d)

def mytiles(info, a):

    x = info[0]
    xdot = info[1]
    return tiles(iht, dim, [8*x/(0.5+1.2), 8*xdot/(0.07+0.07)], [a])

def sarsa_l(alpha, flag):

    gamma = 1
    l_set = [0, 0.9]
    episodes = 50
    runs = 5

    for l in l_set:

        step_counts = np.zeros(len(alpha))

        for p in range(len(alpha)):

            for k in range(runs):

                w = np.zeros(d)
                step_count = 0

                for i in range(episodes):

                    s = env.reset()
                    z = np.zeros(d)

                    sums = []
                    for a_ in range(3):
                        indices = mytiles(s, a_)
                        sum = 0
                        for j in range(dim):
                            sum += w[indices[j]]
                        sums.append(sum)
                    sums = np.array(sums)
                    actions = np.argwhere(sums == np.max(sums))
                    a = actions[np.random.randint(len(actions))].item()

                    while True:

                        info = env.step(a)
                        step_count += 1
                        delta = info[1]
                        indices = mytiles(s, a)
                        for j in range(dim):
                            delta -= w[indices[j]]
                            if flag:
                                z[indices[j]] = 1
                            else:
                                z[indices[j]] += 1
                        if info[2]:
                            w += alpha[p] * delta * z
                            break
                        s = info[0]

                        sums = []
                        for a_ in range(3):
                            indices = mytiles(s, a_)
                            sum = 0
                            for j in range(dim):
                                sum += w[indices[j]]
                            sums.append(sum)
                        sums = np.array(sums)
                        actions = np.argwhere(sums == np.max(sums))
                        a = actions[np.random.randint(len(actions))].item()

                        indices = mytiles(s, a)
                        for j in range(dim):
                            delta += gamma * w[indices[j]]
                        w += alpha[p] * delta * z
                        z *= gamma * l

                step_counts[p] += step_count / runs / episodes

        plt.plot(alpha * 8, step_counts, label = 'lambda = ' + str(l))

    plt.xlabel('alpha * 8')
    plt.ylabel('steps per episode')
    if flag:
        plt.title('Sarsa(lambda) with replacing traces')
    else:
        plt.title('Sarsa(lambda) with accumulating traces')
    plt.legend()
    plt.show()

alpha_1 = np.arange(0.6, 2.0, 0.2) / 8
sarsa_l(alpha_1, True)
alpha_2 = np.arange(0.2, 0.5, 0.05) / 8
sarsa_l(alpha_2, False)