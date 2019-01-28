import numpy as np
import matplotlib.pyplot as plt
import random

arms = np.array([0.1, 0.275, 0.45, 0.625, 0.8])
eps_Greedy = np.array([0.01, 0.1, 0.3])
Q_OptIntVal = np.array([1, 5, 50])
c_UCB = np.array([0.2, 1, 2])

run_count = 2000
time_count = 1000

cum_reg = np.zeros((3,time_count))
avg_rwd = np.zeros((3,time_count))
opt_act = np.zeros((3,time_count))

for i in range(3):

    for j in range(run_count):

        random.seed()

        R_sum = 0
        Q = np.zeros(5)
        N = np.zeros(5)

        for k in range(time_count):

            if eps_Greedy[i] < random.uniform(0,1):
                A = np.argmax(Q)
            else:
                A = random.randint(0,4)
            if arms[A] > random.uniform(0,1):
                R = 1
            else:
                R = 0

            N[A] += 1
            Q[A] = Q[A] + (R-Q[A])/N[A]
            R_sum += R
            avg_rwd[i][k] += R_sum/(k+1)/run_count
            opt_act[i][k] += N[4]/(k+1)/run_count
            cum_reg[i][k] += ((k+1)*arms[4]-R_sum)/run_count

plt.subplot(3,3,1)
plt.plot(range(time_count), cum_reg[0], color = 'r', label='epsilon = 0.01')
plt.plot(range(time_count), cum_reg[1], color = 'b', label='epsilon = 0.1')
plt.plot(range(time_count), cum_reg[2], color = 'g', label='epsilon = 0.3')
plt.xlabel('time step')
plt.ylabel('Cumulative regret')
plt.ylim(0,300)
plt.legend(loc='upper left', fontsize = 8)

plt.subplot(3,3,2)
plt.plot(range(time_count), avg_rwd[0], color = 'r', label='epsilon = 0.01')
plt.plot(range(time_count), avg_rwd[1], color = 'b', label='epsilon = 0.1')
plt.plot(range(time_count), avg_rwd[2], color = 'g', label='epsilon = 0.3')
plt.xlabel('time step')
plt.ylabel('Averaged reward')
plt.ylim(0,1)
plt.legend(loc='upper left', fontsize = 8)

plt.subplot(3,3,3)
plt.plot(range(time_count), opt_act[0], color = 'r', label='epsilon = 0.01')
plt.plot(range(time_count), opt_act[1], color = 'b', label='epsilon = 0.1')
plt.plot(range(time_count), opt_act[2], color = 'g', label='epsilon = 0.3')
plt.xlabel('time step')
plt.ylabel('% Optimal action')
plt.ylim(0,1)
plt.legend(loc='upper left', fontsize = 8)

cum_reg = np.zeros((3,time_count))
avg_rwd = np.zeros((3,time_count))
opt_act = np.zeros((3,time_count))

for i in range(3):

    for j in range(run_count):

        random.seed()

        R_sum = 0
        Q = np.zeros(5)
        Q += Q_OptIntVal[i]
        N = np.zeros(5)

        for k in range(time_count):

            A = np.argmax(Q)
            if arms[A] > random.uniform(0,1):
                R = 1
            else:
                R = 0

            N[A] += 1
            Q[A] = Q[A] + (R-Q[A])/N[A]
            R_sum += R
            avg_rwd[i][k] += R_sum/(k+1)/run_count
            opt_act[i][k] += N[4]/(k+1)/run_count
            cum_reg[i][k] += ((k+1)*arms[4]-R_sum)/run_count

plt.subplot(3,3,4)
plt.plot(range(time_count), cum_reg[0], color = 'r', label = 'Q1 = 1')
plt.plot(range(time_count), cum_reg[1], color = 'b', label = 'Q1 = 5')
plt.plot(range(time_count), cum_reg[2], color = 'g', label = 'Q1 = 50')
plt.xlabel('time step')
plt.ylabel('Cumulative regret')
plt.ylim(0,300)
plt.legend(loc='upper left', fontsize = 8)

plt.subplot(3,3,5)
plt.plot(range(time_count), avg_rwd[0], color = 'r', label = 'Q1 = 1')
plt.plot(range(time_count), avg_rwd[1], color = 'b', label = 'Q1 = 5')
plt.plot(range(time_count), avg_rwd[2], color = 'g', label = 'Q1 = 50')
plt.xlabel('time step')
plt.ylabel('Averaged reward')
plt.ylim(0,1)
plt.legend(loc='upper left', fontsize = 8)

plt.subplot(3,3,6)
plt.plot(range(time_count), opt_act[0], color = 'r', label = 'Q1 = 1')
plt.plot(range(time_count), opt_act[1], color = 'b', label = 'Q1 = 5')
plt.plot(range(time_count), opt_act[2], color = 'g', label = 'Q1 = 50')
plt.xlabel('time step')
plt.ylabel('% Optimal action')
plt.ylim(0,1)
plt.legend(loc='upper left', fontsize = 8)

cum_reg = np.zeros((3,time_count))
avg_rwd = np.zeros((3,time_count))
opt_act = np.zeros((3,time_count))
eps = 1e-10

for i in range(3):

    for j in range(run_count):

        random.seed()

        R_sum = 0
        Q = np.zeros(5)
        N = np.zeros(5)

        for k in range(time_count):

            A = np.argmax(Q+c_UCB[i]*np.sqrt(np.log(k+1)/(N+eps))) # just to avoid something divided by 0, and it affects basically nothing to the answer in future iteration
            if arms[A] > random.uniform(0,1):
                R = 1
            else:
                R = 0

            N[A] += 1
            Q[A] = Q[A] + (R-Q[A])/N[A]
            R_sum += R
            avg_rwd[i][k] += R_sum/(k+1)/run_count
            opt_act[i][k] += N[4]/(k+1)/run_count
            cum_reg[i][k] += ((k+1)*arms[4]-R_sum)/run_count

plt.subplot(3,3,7)
plt.plot(range(time_count), cum_reg[0], color = 'r', label = 'c = 0.2')
plt.plot(range(time_count), cum_reg[1], color = 'b', label = 'c = 1')
plt.plot(range(time_count), cum_reg[2], color = 'g', label = 'c = 2')
plt.xlabel('time step')
plt.ylabel('Cumulative regret')
plt.ylim(0,300)
plt.legend(loc='upper left', fontsize = 8)

plt.subplot(3,3,8)
plt.plot(range(time_count), avg_rwd[0], color = 'r', label = 'c = 0.2')
plt.plot(range(time_count), avg_rwd[1], color = 'b', label = 'c = 1')
plt.plot(range(time_count), avg_rwd[2], color = 'g', label = 'c = 2')
plt.xlabel('time step')
plt.ylabel('Averaged reward')
plt.ylim(0,1)
plt.legend(loc='upper left', fontsize = 8)

plt.subplot(3,3,9)
plt.plot(range(time_count), opt_act[0], color = 'r', label = 'c = 0.2')
plt.plot(range(time_count), opt_act[1], color = 'b', label = 'c = 1')
plt.plot(range(time_count), opt_act[2], color = 'g', label = 'c = 2')
plt.xlabel('time step')
plt.ylabel('% Optimal action')
plt.ylim(0,1)
plt.legend(loc='upper left', fontsize = 8)

plt.show()