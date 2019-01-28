import numpy as np

uniform_policy = np.ones((25,4))/4

def plot_policy(policy):

    for i in range(5):
        row = ''
        for j in range(5):
            count = 0
            if policy[5*i+j,0] > 0:
                row += ('n')
                count += 1
            if policy[5*i+j,1] > 0:
                row += ('e')
                count += 1
            if policy[5*i+j,2] > 0:
                row += ('s')
                count += 1
            if policy[5*i+j,3] > 0:
                row += ('w')
                count += 1
            for k in range(5-count):
                row += (' ')
        print(row)

def gridworld(slip_prob = 0.2):

    P = {}

    for s in range(25):

        P[s] = {}

        for a in range(4):

            P[s][a] = []

            if s == 1:
                P[s][a].append((1 - slip_prob, s + 20, 10))
                P[s][a].append((slip_prob, s + 20, 10))

            elif s == 3:
                P[s][a].append((1 - slip_prob, s + 10, 5))
                P[s][a].append((slip_prob, s + 10, 5))

            else:

                if a == 0:
                    if s < 5:
                        P[s][a].append((1 - slip_prob, s, -1))
                        P[s][a].append((slip_prob, s, -1))
                    else:
                        P[s][a].append((1 - slip_prob, s - 5, 0))
                        P[s][a].append((slip_prob, s, 0))

                if a == 1:
                    if s % 5 == 4:
                        P[s][a].append((1 - slip_prob, s, -1))
                        P[s][a].append((slip_prob, s, -1))
                    else:
                        P[s][a].append((1 - slip_prob, s + 1, 0))
                        P[s][a].append((slip_prob, s, 0))

                if a == 2:
                    if s > 19:
                        P[s][a].append((1 - slip_prob, s, -1))
                        P[s][a].append((slip_prob, s, -1))
                    else:
                        P[s][a].append((1 - slip_prob, s + 5, 0))
                        P[s][a].append((slip_prob, s, 0))

                if a == 3:
                    if s % 5 == 0:
                        P[s][a].append((1 - slip_prob, s, -1))
                        P[s][a].append((slip_prob, s, -1))
                    else:
                        P[s][a].append((1 - slip_prob, s - 1, 0))
                        P[s][a].append((slip_prob, s, 0))

    return P

def policy_eval(P, policy = uniform_policy, theta = 0.0001, gamma = 0.9):

    V = np.zeros((25,1))
    delta = 1

    while(delta >= theta):

        delta = 0

        for s in range(25):

            v = V[s]*1
            V[s] = 0

            for a in range(4):
                
                if P[s][a][0][1] == s:
                    tmp_1 = v
                else:
                    tmp_1 = V[P[s][a][0][1]]
                if P[s][a][1][1] == s:
                    tmp_2 = v
                else:
                    tmp_2 = V[P[s][a][1][1]]
                                        
                V[s] += policy[s, a] * (
                            P[s][a][0][0] * (P[s][a][0][2] + gamma * tmp_1) + P[s][a][1][0] * (
                                P[s][a][1][2] + gamma * tmp_2))
                
            delta = max(delta, abs(v - V[s]))

    V = np.around(V, decimals = 2)

    return V

def policy_iter(P, theta = 0.0001, gamma = 0.9):

    policy = uniform_policy
    tmp_V = np.zeros(4)

    while(True):

        V = policy_eval(P, policy = policy, theta = theta, gamma = gamma)

        err = np.zeros(4)

        for s in range(25):

            A = policy[s]*1

            for a in range(4):

                tmp = P[s][a][0][1]
                tmp_V[a] = P[s][a][0][0] * (P[s][a][0][2] + gamma * V[tmp]) + P[s][a][1][0] * (
                        P[s][a][1][2] + gamma * V[s])

            policy[s] = 0
            act = np.argwhere(tmp_V == np.amax(tmp_V))
            policy[s][act] = 1 / len(act)
            err += abs(policy[s] - A)
        
        if np.linalg.norm(err) < 0.001:
            return V, policy

def value_iter(P, theta = 0.0001, gamma = 0.9):

    V = np.zeros((25, 1))
    delta = 1
    tmp_V = np.zeros(4)

    while (delta >= theta):

        delta = 0

        for s in range(25):

            v = V[s]*1            

            for a in range(4):
                    
                tmp_V[a] = P[s][a][0][0] * (P[s][a][0][2] + gamma * V[P[s][a][0][1]]) + P[s][a][1][0] * (
                        P[s][a][1][2] + gamma * V[P[s][a][1][1]])

            V[s] = np.max(tmp_V)
            delta = max(delta, abs(v - V[s]))
            
    policy = np.zeros((25,4))
            
    for s in range(25):

        for a in range(4):

            tmp_V[a] = P[s][a][0][0] * (P[s][a][0][2] + gamma * V[P[s][a][0][1]]) + P[s][a][1][0] * (
                    P[s][a][1][2] + gamma * V[P[s][a][1][1]])

        act = np.argwhere(tmp_V == np.amax(tmp_V))
        policy[s][act] = 1 / len(act)
    
    V = np.around(V, decimals = 2)
        
    return V, policy

def gridworld_ep(slip_prob = 0.2):

    P = {}

    for s in range(25):

        P[s] = {}

        for a in range(4):

            P[s][a] = []

            if s == 1:
                P[s][a].append((1 - slip_prob, s + 20, 10))
                P[s][a].append((slip_prob, s + 20, 10))

            elif s == 3:
                P[s][a].append((1 - slip_prob, s + 10, 5))
                P[s][a].append((slip_prob, s + 10, 5))

            elif s == 21 or s == 13:
                P[s][a].append((1 - slip_prob, s, 0))
                P[s][a].append((slip_prob, s, 0))

            else:

                if a == 0:
                    if s < 5:
                        P[s][a].append((1 - slip_prob, s, -1))
                        P[s][a].append((slip_prob, s, -1))
                    else:
                        P[s][a].append((1 - slip_prob, s - 5, 0))
                        P[s][a].append((slip_prob, s, 0))

                if a == 1:
                    if s % 5 == 4:
                        P[s][a].append((1 - slip_prob, s, -1))
                        P[s][a].append((slip_prob, s, -1))
                    else:
                        P[s][a].append((1 - slip_prob, s + 1, 0))
                        P[s][a].append((slip_prob, s, 0))

                if a == 2:
                    if s > 19:
                        P[s][a].append((1 - slip_prob, s, -1))
                        P[s][a].append((slip_prob, s, -1))
                    else:
                        P[s][a].append((1 - slip_prob, s + 5, 0))
                        P[s][a].append((slip_prob, s, 0))

                if a == 3:
                    if s % 5 == 0:
                        P[s][a].append((1 - slip_prob, s, -1))
                        P[s][a].append((slip_prob, s, -1))
                    else:
                        P[s][a].append((1 - slip_prob, s - 1, 0))
                        P[s][a].append((slip_prob, s, 0))

    return P

P = gridworld()
V = policy_eval(P)
print(V.reshape(5,5))
V, policy = policy_iter(P)
print(V.reshape(5,5))
plot_policy(policy)
V, policy = value_iter(P)
print(V.reshape(5,5))
plot_policy(policy)

P_ep = gridworld_ep()
V, policy = value_iter(P_ep, gamma = 1)
print(V.reshape(5,5))
plot_policy(policy)
V, policy = value_iter(P_ep, gamma = 0.9)
print(V.reshape(5,5))
plot_policy(policy)
V, policy = value_iter(P_ep, gamma = 0.8)
print(V.reshape(5,5))
plot_policy(policy)
V, policy = value_iter(P_ep, gamma = 0.7)
print(V.reshape(5,5))
plot_policy(policy)
