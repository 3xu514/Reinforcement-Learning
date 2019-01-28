import numpy as np
import matplotlib.pyplot as plt

gamma = 0.9

def gridworld(goal = 1000):

    P = {}

    for s_tmp in range(25):

        s = s_tmp
        P[s] = {}

        for a in range(4):

            if a == 0:
                if s_tmp < 5:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s - 5, 0]

            if a == 1:
                if s_tmp % 5 == 4:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s + 1, 0]

            if a == 2:
                if s_tmp > 19:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s + 5, 0]

            if a == 3:
                if s_tmp % 5 == 0:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s - 1, 0]

    for s_tmp in range(30):

        s = s_tmp + 25
        P[s] = {}

        for a in range(4):

            if a == 0:
                if s_tmp < 5:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s - 5, 0]

            if a == 1:
                if s_tmp % 5 == 4:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s + 1, 0]

            if a == 2:
                if s_tmp > 24:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s + 5, 0]

            if a == 3:
                if s_tmp % 5 == 0:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s - 1, 0]

    for s_tmp in range(25):

        s = s_tmp + 55
        P[s] = {}

        for a in range(4):

            if a == 0:
                if s_tmp < 5:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s - 5, 0]

            if a == 1:
                if s_tmp % 5 == 4:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s + 1, 0]

            if a == 2:
                if s_tmp > 19:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s + 5, 0]

            if a == 3:
                if s_tmp % 5 == 0:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s - 1, 0]

    for s_tmp in range(20):

        s = s_tmp + 80
        P[s] = {}

        for a in range(4):

            if a == 0:
                if s_tmp < 5:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s - 5, 0]

            if a == 1:
                if s_tmp % 5 == 4:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s + 1, 0]

            if a == 2:
                if s_tmp > 14:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s + 5, 0]

            if a == 3:
                if s_tmp % 5 == 0:
                    P[s][a] = [s, 0]
                else:
                    P[s][a] = [s - 1, 0]

    P[100] = {}
    P[100][0] = [21, 0]
    P[21][2] = [100, 0]
    P[100][1] = [100, 0]
    P[100][2] = [56, 0]
    P[56][0] = [100, 0]
    P[100][3] = [100, 0]
    P[101] = {}
    P[101][0] = [101, 0]
    P[101][1] = [35, 0]
    P[35][3] = [101, 0]
    P[101][2] = [101, 0]
    P[101][3] = [14, 0]
    P[14][1] = [101, 0]
    P[102] = {}
    P[102][0] = [52, 0]
    P[52][2] = [102, 0]
    P[102][1] = [102, 0]
    P[102][2] = [82, 0]
    P[82][0] = [102, 0]
    P[102][3] = [102, 0]
    P[103] = {}
    P[103][0] = [103, 0]
    P[103][1] = [90, 0]
    P[90][3] = [103, 0]
    P[103][2] = [103, 0]
    P[103][3] = [74, 0]
    P[74][1] = [103, 0]

    if goal != 1000:

        P[goal][0] = [goal, 0]
        P[goal][1] = [goal, 0]
        P[goal][2] = [goal, 0]
        P[goal][3] = [goal, 0]
        for s in range(104):
            for a in range(4):
                if P[s][a][0] == goal and s!= goal:
                    P[s][a][1] = 1

    return P

def value_iter(P, iter = 100, goal = 102):

    V = np.zeros((104, 1))
    tmp_V = np.zeros(4)

    for it in range(iter):

        V_prep = V*1

        for s in range(104):

            for a in range(4):
                tmp_V[a] = 2/3 * (P[s][a][1] + gamma * V_prep[P[s][a][0]])
                for a_ in range(4):
                    if a_ != a:
                        tmp_V[a] += 1/9 * (P[s][a_][1] + gamma * V_prep[P[s][a_][0]])
            if s != goal:
                V[s] = np.max(tmp_V)

    p = np.zeros((104, 1))
    V[goal] = 1

    for s in range(104):

        for a in range(4):
            tmp_V[a] = 2/3 * (P[s][a][1] + gamma * V[P[s][a][0]])
            for a_ in range(4):
                if a_ != a:
                    tmp_V[a] += 1/9 * (P[s][a_][1] + gamma * V[P[s][a_][0]])

        p[s] = np.argmax(tmp_V)

    V = np.around(V, decimals=2)

    return V, p

def options(p, p_r):

    p[100] = 2
    p[102] = 0
    p_r[100] = 0
    p_r[102] = 2

    P = gridworld()
    p_o1 = np.zeros((104, 2))
    p_o2 = np.zeros((104, 2))
    p_mat = np.zeros((104, 104))
    p_mat_r = np.zeros((104, 104))

    for s in range(104):
        for a in range(4):
            s_ = P[s][a][0]
            if a == p[s]:
                p_mat[s][s_] = 2/3
            else:
                p_mat[s][s_] = 1/9
            if a == p_r[s]:
                p_mat_r[s][s_] = 2/3
            else:
                p_mat_r[s][s_] = 1/9

    opt = np.zeros((104, 2))
    for s_tmp in range(25):
        s = s_tmp
        opt[s] = [101, 100]
    for s_tmp in range(30):
        s = s_tmp + 25
        opt[s] = [102, 101]
    for s_tmp in range(25):
        s = s_tmp + 55
        opt[s] = [103, 100]
    for s_tmp in range(20):
        s = s_tmp + 80
        opt[s] = [102, 103]
    opt[100] = [103, 101]
    opt[101] = [102, 100]
    opt[102] = [101, 103]
    opt[103] = [102, 100]

    p_m = p_mat*1
    p_mr = p_mat_r*1
    p_mat[100:104] = 0
    p_mat_r[100:104] = 0
    for i in range(1000):
        for s in range(100):
            for o in range(2):
                p_o1[s, o] += gamma ** (i + 1) * p_m[s][int(opt[s, o])]
                p_o2[s, o] += gamma ** (i + 1) * p_mr[s][int(opt[s, o])]
        p_m = np.dot(p_m, p_mat)
        p_mr = np.dot(p_mr, p_mat_r)

    p_o1[100, 0] = p_o1[56, 0] * 3 / 4
    p_o1[100, 1] = p_o1[21, 1] * 1 / 8
    p_o2[100, 0] = p_o2[21, 0] * 3 / 4
    p_o2[100, 1] = p_o2[56, 1] * 1 / 8
    p_o1[101, 0] = p_o1[35, 0] * 3 / 4
    p_o1[101, 1] = p_o1[14, 1] * 1 / 8
    p_o2[101, 0] = p_o2[14, 0] * 3 / 4
    p_o2[101, 1] = p_o2[35, 1] * 1 / 8
    p_o1[102, 0] = p_o1[52, 0] * 3 / 4
    p_o1[102, 1] = p_o1[82, 1] * 1 / 8
    p_o2[102, 0] = p_o2[82, 0] * 3 / 4
    p_o2[102, 1] = p_o2[52, 1] * 1 / 8
    p_o1[103, 0] = p_o1[90, 0] * 3 / 4
    p_o1[103, 1] = p_o1[74, 1] * 1 / 8
    p_o2[103, 0] = p_o2[74, 0] * 3 / 4
    p_o2[103, 1] = p_o2[90, 1] * 1 / 8

    return p_o1, p_o2, opt

def option_iter(p_o1, p_o2, opt, iter = 100, goal = 102):

    V_o = np.zeros((104, 1))
    V_o[goal] = 1

    for it in range(iter):

        V_oprep = V_o*1

        for s in range(104):
            tmp1 = V_oprep[int(opt[s, 0])] * p_o1[s][0] + V_oprep[int(opt[s, 1])] * p_o1[s][1]
            tmp2 = V_oprep[int(opt[s, 0])] * p_o2[s][0] + V_oprep[int(opt[s, 1])] * p_o2[s][1]
            if s != goal:
                V_o[s] = max(tmp1, tmp2)

    return V_o

def option_action_iter(P, p_o1, p_o2, opt, iter = 100, goal = 87):

    V_oa = np.zeros((104, 1))
    tmp_V = np.zeros(4)

    for it in range(iter):

        V_oaprep = V_oa*1

        for s in range(104):
            tmp1 = V_oaprep[int(opt[s, 0])] * p_o1[s][0] + V_oaprep[int(opt[s, 1])] * p_o1[s][1]
            tmp2 = V_oaprep[int(opt[s, 0])] * p_o2[s][0] + V_oaprep[int(opt[s, 1])] * p_o2[s][1]
            for a in range(4):
                tmp_V[a] = 2/3 * (P[s][a][1] + gamma * V_oaprep[P[s][a][0]])
                for a_ in range(4):
                    if a_ != a:
                        tmp_V[a] += 1/9 * (P[s][a_][1] + gamma * V_oaprep[P[s][a_][0]])
            tmp3 = np.max(tmp_V)
            if s != goal:
                V_oa[s] = np.max([tmp1, tmp2, tmp3])

    V_oa[goal] = 1

    return V_oa

def plot_gridworld(V):

    w = -np.ones((13, 13))
    for i in range(5):
        for j in range(5):
            w[i+1, j+1] = V[i*5 + j]
    for i in range(6):
        for j in range(5):
            w[i+1, j+7] = V[25 + i*5 + j]
    for i in range(5):
        for j in range(5):
            w[i+7, j+1] = V[55 + i*5 + j]
    for i in range(4):
        for j in range(5):
            w[i+8, j+7] = V[80 + i*5 + j]
    w[6, 2] = V[100]
    w[3, 6] = V[101]
    w[7, 9] = V[102]
    w[10, 6] = V[103]
    plt.imshow(w, cmap = plt.get_cmap('hot'))
    plt.colorbar()
    plt.xticks([])
    plt.yticks([])

    return

P = gridworld(goal = 102)
P_r = gridworld(goal = 100)
V, p = value_iter(P, iter = 100, goal = 102)
V_r, p_r = value_iter(P_r, iter = 100, goal = 100)
p_o1, p_o2, opt = options(p, p_r)

for i in range(3):
    V, p = value_iter(P, iter = i, goal = 102)
    plt.subplot(2, 3, i + 1)
    plot_gridworld(V)
for i in range(3):
    V_o = option_iter(p_o1, p_o2, opt, iter = i, goal = 102)
    plt.subplot(2, 3, i + 4)
    plot_gridworld(V_o)
plt.show()

for i in range(6):
    V_oa = option_action_iter(gridworld(goal = 87), p_o1, p_o2, opt, iter = i, goal = 87)
    plt.subplot(2, 3, i + 1)
    plot_gridworld(V_oa)
plt.show()