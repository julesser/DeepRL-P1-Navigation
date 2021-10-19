from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

from dqn_agent import Agent


def dqn(n_episodes=500, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """Deep Q-Learning.

    Params
    ======
        n_episodes (int): maximum number of training episodes
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    avgs = []                          # average returns
    eps = eps_start                    # initialize epsilon
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[
            brain_name]  # reset the environment
        # get the current state
        state = env_info.vector_observations[0]
        score = 0                                          # initialize the score
        while True:
            # action = np.random.randint(action_size)
            action = agent.act(state, eps)
            # send the action to the environment
            env_info = env.step(action)[brain_name]
            # get the next state
            next_state = env_info.vector_observations[0]
            # get the reward
            reward = env_info.rewards[0]
            # see if episode has finished
            done = env_info.local_done[0]
            agent.step(state, action, reward, next_state, done)  # update agent
            # roll over the state to next time step
            state = next_state
            score += reward                                     # update the score
            if done:
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        avg = np.mean(scores_window)      # calculate moving average
        avgs.append(avg)                  # save most recent moving average
        eps = max(eps_end, eps_decay*eps)  # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(
            i_episode, np.mean(scores_window)), end="")
        if i_episode % 50 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(
                i_episode, np.mean(scores_window)))
        if np.mean(scores_window) >= 13.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(
                i_episode-100, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
            break
    return scores, avgs


# 1. Create environment
env = UnityEnvironment(file_name="simulator/Banana.x86_64")
brain_name = env.brain_names[0]  # get default brain
brain = env.brains[brain_name]
env_info = env.reset(train_mode=True)[brain_name]  # reset environment
action_size = brain.vector_action_space_size  # number of actions
state = env_info.vector_observations[0]  # get state space
state_size = len(state)

# 2. Create agent
agent = Agent(state_size, action_size, seed=0)

# 3. Roll out DQN algorithm
scores, avgs = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores, label='DQN')
plt.plot(np.arange(len(scores)), avgs, c='r', label='Average')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend()
plt.show()
