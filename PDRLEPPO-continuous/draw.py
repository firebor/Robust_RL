import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def smooth(reward):
    smooth_reward = []
    for i in range(reward.shape[0]):
        if i == 0:
            smooth_reward.append(reward[i])
        else:
            smooth_reward.append(smooth_reward[-1] * 0.9 + reward[i] * 0.1)
    return np.array(smooth_reward)


env_name = ['BipedalWalker-v3', 'MountainCarContinuous-v0', 'LunarLanderContinuous-v2']
colors = ['r', 'darkorange', 'dodgerblue', 'limegreen', 'yellow', 'magenta', 'chocolate', 'indigo', 'gray', 'aqua', 'g',
          'black']


def get_data(algorithm, env_index, number, ):
    reward1 = smooth(
        np.load('./data_train/{}_discrete_env_{}_number_{}_seed_0.npy'.format(algorithm, env_name[env_index], number)))
    len = reward1.shape[0]
    return reward1, len


def drawing_LL(plt, algorithm, number, color, label):
    reward, len = get_data(algorithm=algorithm, env_index=1, number=number)
    plt.plot(reward, color=color, label=label)
    plt.title("LunarLander-v2", size=14)
    plt.xlabel("Steps", size=14)
    plt.ylabel("Reward", size=14)
    # plt.xticks([0, 20, 40, 60, 80], ['0', '10k', '20k', '30k', '40k'], size=14)
    plt.xticks([0, 40, 80, 120, 160, 200], ['0', '20k', '40k', '60k', '80k', '100k'], size=14)
    plt.yticks(size=14)
    plt.ylim([-300, 300])
    plt.legend(loc='lower right', fontsize=14)


def drawing_DP(plt, algorithm, number, color, label):
    reward, len = get_data(algorithm=algorithm, env_index=1, number=number)
    plt.plot(reward, color=color, label=label)
    plt.title("LunarLander-v2", size=14)
    plt.xlabel("Steps", size=14)
    plt.ylabel("Reward", size=14)
    plt.xticks([0, 40, 80, 120, 160, 200], ['0', '20k', '40k', '60k', '80k', '100k'], size=14)
    plt.yticks(size=14)
    plt.ylim([-300, 300])
    plt.legend(loc='lower right', fontsize=14)


def draw_evaluate(plt, algorithm, method, color, env_index, label):
    reward = np.load('./data_evaluate/{}_env_{}_{}.npy'.format(algorithm, env_name[env_index], method))
    plt.plot(reward, color=color, label=label)
    plt.title("BipedalWalker-v3", size=14)
    plt.xlabel("sigma", size=14)
    plt.ylabel("Reward", size=14)
    plt.xticks([0, 20, 40, 60, 80, 100, 120, 140, 160, 180, 200], size=14)
    plt.yticks(size=14)
    # plt.ylim([-1200, -100])
    # plt.xlim([0, 1.2])
    plt.legend(loc='lower right', fontsize=14)


if __name__ == '__main__':
    # sns.set_style('darkgrid')
    plt.figure()
    # drawing_LL(plt, algorithm='PPO', number=1, color=colors[0], label='PPO')
    # drawing_DP(plt, algorithm='PPODP', number=1, color=colors[1], label='PPODP')
    draw_evaluate(plt, algorithm="PPO_continuous", method="origin", color=colors[0], env_index=0, label="PPO")
    draw_evaluate(plt, algorithm="PPO_continuous", method="DP", color=colors[1], env_index=0, label="PPODP")
    plt.show()
