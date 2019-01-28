import numpy as np
import matplotlib.pyplot as plt

v_true = np.load('trueStateValue.npy')

alpha = np.arange(0, 1.02, 0.02)
n = np.power(2, np.arange(0, 10))
episodes = 10
gamma = 1
runs = 100

s_count = len(v_true)
a_count = 200
agg_count = 20
s_per_agg = int(s_count / agg_count)
d = agg_count + 2

for n_i in n:

    rms_set = []
    print(n_i)

    for alpha_i in alpha:

        rms = 0

        for run in range(runs):

            w = np.zeros((1, d))
            x = np.eye(d)
            for i in range(episodes):

                T = float('Inf')
                s = 501
                t = 0
                r_list = [0]
                s_list = [s]

                while True:

                    if t < T:
                        a = np.random.randint(a_count)
                        if a < 100:
                            s = s - (100 - a)
                        else:
                            s = s + (a - 99)
                        if s < 1:
                            r_list.append(-1)
                            T = t + 1
                            s = 0
                        elif s > 1000:
                            r_list.append(1)
                            T = t + 1
                            s = 1001
                        else:
                            r_list.append(0)
                        s_list.append(s)

                    tao = t - n_i + 1

                    if tao >= 0:

                        G = 0
                        for j in range(tao + 1, min(tao + n_i, T) + 1):
                            G += np.power(gamma, j - tao - 1) * r_list[j]
                        if tao + n_i < T:
                            if s_list[tao + n_i] == 0:
                                s_agg = 0
                            elif s_list[tao + n_i] == 1001:
                                s_agg = d - 1
                            else:
                                s_agg = int((s_list[tao + n_i] - 1) / s_per_agg) + 1
                            G += np.power(gamma, n_i) * np.dot(w, x[:, s_agg])
                        s_agg = int((s_list[tao] - 1) / s_per_agg) + 1
                        w += alpha_i * (G - np.dot(w, x[:, s_agg])) * x[:, s_agg]

                    if tao == T - 1:
                        break
                    t += 1

            tmp = np.dot(w, x).reshape(d)
            err_vec = []
            for i in range(agg_count):
                err_vec.extend(v_true[s_per_agg*i + 1: s_per_agg*(i+1) + 1] - tmp[i + 1])
            rms += (np.linalg.norm(err_vec) / np.sqrt(s_count) - rms) / (run + 1)

        rms_set.append(rms)

    plt.plot(alpha, rms_set, label = 'n = ' + str(n_i))

plt.ylim(0.15, 0.55)
plt.xlabel('alpha')
plt.ylabel('average RMS over 100 runs')
plt.title('RMS of different alpha and n')
plt.legend()
plt.show()