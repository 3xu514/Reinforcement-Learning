import numpy as np
import matplotlib.pyplot as plt

alphas = np.array([4, 2, 1]) / 2**14
runs = 100
episodes = 1000
x = np.eye(2)

np.random.seed(0)

for alpha in alphas:

    G_set = np.zeros(episodes)

    for i in range(runs):

        theta = np.ones((1, 2))
        theta[0][1] = -2

        for j in range(episodes):

            s = 0
            s_set = [s]
            a_set = []

            while True:

                h = np.exp(np.dot(theta, x)) / np.sum(np.exp(np.dot(theta, x)))
                tmp = np.random.rand()
                if tmp > h[0][0]:
                    if s == 1:
                        s += 1
                    elif s == 2:
                        s -= 1
                    a_set.append(1)
                else:
                    if s == 1:
                        s -= 1
                    else:
                        s += 1
                    a_set.append(0)
                if s == 3:
                    break
                s_set.append(s)

            for k in range(len(s_set)):
                G = k - len(s_set)
                theta += alpha * G * (x[:, a_set[k]] - np.sum(
                    np.exp(np.dot(theta, x)) / np.sum(np.exp(np.dot(theta, x))) * x, axis = 1))

            G_set[j] -= len(s_set) / runs

    if alpha == 2**-12:
        plt.plot(range(episodes), G_set, label = 'alpha = $2^{-12}$', color = 'b')
    elif alpha == 2**-13:
        plt.plot(range(episodes), G_set, label = 'alpha = $2^{-13}$', color = 'r')
    else:
        plt.plot(range(episodes), G_set, label = 'alpha = $2^{-14}$', color = 'g')

plt.title('Monte Carlo Policy Gradient Control')
plt.xlabel('episodes')
plt.ylabel('total reward on episode')
plt.legend()
plt.show()

alpha = 2**-13
runs = 100
episodes = 1000
x = np.eye(2)
alpha_theta = 2**-9
alpha_w = 2**-6
x_ = np.eye(3)
G_set = np.zeros(episodes)
G_set_b = np.zeros(episodes)

np.random.seed(41)

for i in range(runs):

    theta = np.ones((1, 2))
    theta[0][1] = -2
    w = np.zeros((1, 3))

    for j in range(episodes):

        s = 0
        s_set = [s]
        a_set = []

        while True:

            h = np.exp(np.dot(theta, x)) / np.sum(np.exp(np.dot(theta, x)))
            tmp = np.random.rand()
            if tmp > h[0][0]:
                if s == 1:
                    s += 1
                elif s == 2:
                    s -= 1
                a_set.append(1)
            else:
                if s == 1:
                    s -= 1
                else:
                    s += 1
                a_set.append(0)
            if s == 3:
                break
            s_set.append(s)

        for k in range(len(s_set)):
            G = k - len(s_set)
            delta = G - np.dot(w, x_[:, s_set[k]])
            w += alpha_w * delta * x_[:, s_set[k]]
            theta += alpha_theta * delta * (x[:, a_set[k]] - np.sum(
                np.exp(np.dot(theta, x)) / np.sum(np.exp(np.dot(theta, x))) * x, axis = 1))

        G_set_b[j] -= len(s_set) / runs

for i in range(runs):

    theta = np.ones((1, 2))
    theta[0][1] = -2

    for j in range(episodes):

        s = 0
        s_set = [s]
        a_set = []

        while True:

            h = np.exp(np.dot(theta, x)) / np.sum(np.exp(np.dot(theta, x)))
            tmp = np.random.rand()
            if tmp > h[0][0]:
                if s == 1:
                    s += 1
                elif s == 2:
                    s -= 1
                a_set.append(1)
            else:
                if s == 1:
                    s -= 1
                else:
                    s += 1
                a_set.append(0)
            if s == 3:
                break
            s_set.append(s)

        for k in range(len(s_set)):
            G = k - len(s_set)
            theta += alpha * G * (x[:, a_set[k]] - np.sum(
                np.exp(np.dot(theta, x)) / np.sum(np.exp(np.dot(theta, x))) * x, axis = 1))

        G_set[j] -= len(s_set) / runs

plt.plot(range(episodes), G_set, label = 'alpha = $2^{-13}$', color = 'r')
plt.plot(range(episodes), G_set_b, label = 'with baseline, alpha_theta = $2^{-9}$ and alpha_w = $2^{-6}$', color = 'g')

plt.title('REINFORCE with Baseline')
plt.xlabel('episodes')
plt.ylabel('total reward on episode')
plt.legend()
plt.show()