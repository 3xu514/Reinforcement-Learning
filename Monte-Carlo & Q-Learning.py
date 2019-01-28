import numpy as np
import matplotlib.pyplot as plt
import gym
import mytaxi

env = gym.make('Taxi-v3').unwrapped
policy = np.load('policy.npy')

def policy_eval(P, policy, theta = 0.01, gamma = 1):

    V = np.zeros((env.nS, 1))
    delta = 1

    while (delta >= theta):

        delta = 0

        V1 = V*1
        V = np.zeros((env.nS, 1))

        for s in range(env.nS):

            for a in range(env.nA):

                V[s] += policy[s, a] * (P[s][a][0][2] + gamma * V1[P[s][a][0][1]])

            delta = max(delta, abs(V1[s] - V[s]))

    V = np.around(V, decimals = 2)

    return V

def mc_prediction(env, policy, baseline, gamma = 1, episodes = 50000):

    np.random.seed(3)
    env.seed(5)
    V = np.zeros((env.nS, 1))
    returns = {}
    rms = []

    for j in range(episodes):

        G = 0
        s = env.reset()
        generated_states = []
        reward = []

        while True:

            act = np.random.rand()
            prob = 0
            for i in range(env.nA):
                prob += policy[s][i]
                if prob > act:
                    action = i
                    break

            info = env.step(action)

            if s not in generated_states:
                reward.append([info[1], True])
            else:
                reward.append([info[1], False])

            generated_states.append(s)
            s = info[0]

            if info[2]:
                break

        for i in reversed(range(len(generated_states))):

            G = gamma * G + reward[i][0]
            if reward[i][1]:
                if generated_states[i] in returns:
                    returns[generated_states[i]].append(G)
                else:
                    returns[generated_states[i]] = [G]
                V[generated_states[i]] = np.average(returns[generated_states[i]])

        rms.append(np.sqrt(sum((V-baseline)**2)/len(V)))

    return rms, V

def mc_control(env, epsilon = 0.1, gamma = 1, episodes = 10000, runs = 10, T = 1000):

    np.random.seed(3)
    env.seed(5)
    avgrew = np.zeros(int(episodes/50))

    for k in range(runs):

        Q = np.zeros((env.nS, env.nA))
        policy = np.ones((env.nS, env.nA))/env.nA
        returns = {}

        for j in range(episodes):

            G = 0
            t = 1
            s = env.reset()
            generated_states_actions = []
            reward = []

            while True:

                t += 1

                act = np.random.rand()
                prob = 0
                for i in range(env.nA):
                    prob += policy[s][i]
                    if prob > act:
                        action = i
                        break

                info = env.step(action)

                if env.nA*s+action not in generated_states_actions:
                    reward.append([info[1], True])
                else:
                    reward.append([info[1], False])

                generated_states_actions.append(env.nA*s+action)

                s = info[0]

                if info[2] or t > T:
                    break

            for i in reversed(range(len(generated_states_actions))):

                G = gamma * G + reward[i][0]
                if reward[i][1]:
                    s = int(generated_states_actions[i] / env.nA)
                    act = generated_states_actions[i] % env.nA
                    if generated_states_actions[i] in returns:
                        returns[generated_states_actions[i]].append(G)
                    else:
                        returns[generated_states_actions[i]] = [G]

                    Q[s][act] = np.average(returns[generated_states_actions[i]])
                    # a = np.argmax(Q[s])
                    tmp = np.argwhere(Q[s] == max(Q[s]))
                    a = int(tmp[np.random.randint(len(tmp))].squeeze())
                    policy[s][:] = epsilon / env.nA
                    policy[s][a] = 1 - epsilon * (env.nA - 1) / env.nA

            if j % 50 == 0:
                avgrew[int(j/50)] += G/runs

    return avgrew

def td0(env, policy, baseline, gamma = 1, alpha = 0.1, episodes = 50000):

    np.random.seed(3)
    env.seed(5)
    V = np.zeros((env.nS, 1))
    rms = []

    for j in range(episodes):

        s = env.reset()

        while True:

            act = np.random.rand()
            prob = 0
            for i in range(env.nA):
                prob += policy[s][i]
                if prob > act:
                    action = i
                    break

            info = env.step(action)
            V[s] += alpha*(info[1] + gamma*V[info[0]] - V[s])
            s = info[0]

            if info[2]:
                break

        rms.append(np.sqrt(sum((V - baseline) ** 2) / len(V)))

    return rms, V

def qlearn(env, gamma = 1, alpha = 0.9, epsilon = 0.1, runs = 10, episodes = 500):

    np.random.seed(3)
    env.seed(5)
    avgrew = np.zeros(episodes)

    for k in range(runs):

        Q = np.zeros((env.nS, env.nA))

        for j in range(episodes):

            s = env.reset()
            G = 0

            while True:

                greedy = np.random.rand()
                if greedy < epsilon:
                    action = np.random.randint(env.nA)
                else:
                    tmp = np.argwhere(Q[s] == max(Q[s]))
                    action = int(tmp[np.random.randint(len(tmp))].squeeze())

                info = env.step(action)
                G += info[1]
                Q[s][action] += alpha*(info[1] + gamma*max(Q[info[0]]) - Q[s][action])
                s = info[0]

                if info[2]:
                    break

            avgrew[j] += G/runs

    return avgrew

def plot_bd(V, rms_mc, V_mc, rms_td0, V_td0):

    plt.figure(figsize=(20, 5))
    plt.style.use('ggplot')

    plt.subplot(141)
    plt.plot(range(len(rms_mc)), rms_mc, color='r', ls='-')
    plt.title('MC prediction')
    plt.xlabel('Episodes')
    plt.ylabel('RMS')

    plt.subplot(142)
    plt.scatter(range(env.nS), V_mc, color='r', marker='x')
    plt.scatter(range(env.nS), V, color='', edgecolors='b', marker='o')
    plt.title('MC prediction')
    plt.xlabel('State')
    plt.ylabel('$V_π(s)$')

    plt.subplot(143)
    plt.plot(range(len(rms_td0)), rms_td0, color='r', ls='-')
    plt.title('TD0')
    plt.xlabel('Episodes')
    plt.ylabel('RMS')

    plt.subplot(144)
    plt.scatter(range(env.nS), V_td0, color='r', marker='x')
    plt.scatter(range(env.nS), V, color='', edgecolors='b', marker='o')
    plt.title('TD0')
    plt.xlabel('State')
    plt.ylabel('$V_π(s)$')

    plt.show() # for Pycharm

def plot_ce(avgrew_mc, avgrew_q):

    plt.figure(figsize=(10, 5))
    plt.style.use('ggplot')

    plt.subplot(121)
    plt.plot(range(0, len(avgrew_mc) * 50, 50), avgrew_mc, color='r', ls='-')
    plt.title('MC Control')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards received within each episode')

    plt.subplot(122)
    plt.plot(range(len(avgrew_q)), avgrew_q, color='r', ls='-')
    plt.title('Q learning')
    plt.xlabel('Episodes')
    plt.ylabel('Sum of rewards received within each episode')

    plt.show() # for Pycharm

V = policy_eval(env.P, policy)

rms_mc, V_mc = mc_prediction(env, policy, V)
rms_td0, V_td0 = td0(env, policy, V)
plot_bd(V, rms_mc, V_mc, rms_td0, V_td0)

avgrew_mc = mc_control(env)
avgrew_q = qlearn(env)
plot_ce(avgrew_mc, avgrew_q)