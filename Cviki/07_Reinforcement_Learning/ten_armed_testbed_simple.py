#######################################################################
# Copyright (C)                                                       #
# 2016-2018 Shangtong Zhang(zhangshangtong.cpp@gmail.com)             #
# 2016 Tian Jun(tianjun.cpp@gmail.com)                                #
# 2016 Artem Oboturov(oboturov@gmail.com)                             #
# 2016 Kenta Shimada(hyperkentakun@gmail.com)                         #
# Permission given to modify the code as long as you keep this        #
# declaration at the top                                              #
#######################################################################

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

import os
print(os.getcwd()) 

class Bandit:

    # @k_arm: # of arms
    # @epsilon: probability for exploration in epsilon-greedy algorithm
    # @initial: initial estimation for each action
    # @step_size: constant step size for updating estimations


    def __init__(self, k_arm=10, epsilon=0., initial=0., step_size=0.1,  true_reward=0., exploration_dec=False):
        self.k = k_arm
        self.step_size = step_size
        self.indices = np.arange(self.k)
        self.time = 0
        self.average_reward = 0
        self.true_reward = true_reward
        self.epsilon = epsilon
        self.initial = initial
        self.exploration = exploration_dec

    def setEpsilon(self, constant):
        self.epsilon = self.epsilon * constant

    def getExploration(self):
        return self.exploration

    def reset(self):
        # real reward for each action
        self.q_true = np.random.randn(self.k) + self.true_reward

        # estimation for each action
        self.q_estimation = np.zeros(self.k) + self.initial

        self.action_count = np.zeros(self.k)
        self.best_action = np.argmax(self.q_true)


    # get an action for this bandit
    def act(self):
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.indices)

        return np.argmax(self.q_estimation)

    # take an action, update estimation for this action
    def step(self, action):
        # generate the reward under N(real reward, 1)
        reward = np.random.randn() + self.q_true[action]
        self.time += 1
        self.average_reward = (self.time - 1.0) / self.time * self.average_reward + reward / self.time
        self.action_count[action] += 1

        self.q_estimation[action] += self.step_size * (reward - self.q_estimation[action])
        return reward

def simulate(runs, time, bandits):
    best_action_counts= np.zeros((len(bandits), runs, time))
    rewards = np.zeros(best_action_counts.shape)
    for i, bandit in enumerate(bandits):
        for r in tqdm(range(runs)):
            bandit.reset()
            if bandit.exploration == True:
                bandit.setEpsilon(0.999)
                #bandit.epsilon *= 0.999
            for t in range(time):
                action = bandit.act()
                reward = bandit.step(action)
                rewards[i, r, t] = reward
                if action == bandit.best_action:
                    best_action_counts[i, r, t] = 1


    best_action_counts = best_action_counts.mean(axis = 1 )
    rewards = rewards.mean(axis = 1)

    return best_action_counts, rewards

def figure_2_1():
    plt.violinplot(dataset=np.random.randn(200,10) + np.random.randn(10))
    plt.xlabel("Action")
    plt.ylabel("Reward distribution")
    if not os.path.exists('./images'):
        os.makedirs('./images')
    plt.savefig('./images/figure_2_1.png')
    plt.close()

def figure_2_2(runs=2000, time=4000):
    epsilons = [0, 0.1, 0.01, 0.2]
    bandits = []

    for eps in epsilons:
        bandits.append(Bandit(epsilon=eps))
        
    bandits.append(Bandit(epsilon=0.1, exploration_dec=True))

    best_action_counts, rewards = simulate(runs, time, bandits)

    plt.figure(figsize=(10, 20))

    plt.subplot(2, 1, 1)
    
    eps_labels = ['epsilon = %.02f' % eps for eps in epsilons]
    # append plot label for the last curve with agent that has decreasing epsilon
    eps_labels.append('epsilon decreasing, starting at 0.1')
    
    for eps_label, rewards in zip(eps_labels, rewards):
        plt.plot(rewards, label=eps_label)

    plt.xlabel('steps')
    plt.ylabel('average reward')
    plt.legend()

    plt.subplot(2, 1, 2)
    for eps_label, counts in zip(eps_labels, best_action_counts):
        plt.plot(counts, label=eps_label)
        
    plt.xlabel('steps')
    plt.ylabel('% optimal action')
    plt.legend()

    if not os.path.exists('./images'):
        os.makedirs('./images')
    plt.savefig('./images/figure_2_2.png')
    plt.close()

if __name__ == '__main__':
    figure_2_1()
    figure_2_2()
