from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import torch

from dqn_agent import Agent

# 1. Create environment
env = UnityEnvironment(file_name="simulator/Banana.x86_64")
brain_name = env.brain_names[0]  # get default brain
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]  # reset environment
action_size = brain.vector_action_space_size  # number of actions
state = env_info.vector_observations[0]  # get state space
state_size = len(state)

# 2. Create agent
agent = Agent(state_size, action_size, seed=0)

# 3. Load the previously learned weights from file
agent.qnetwork_local.load_state_dict(torch.load('checkpoint.pth'))

# 4. Apply trained agent to solve three episodes
for i in range(3):
    env_info = env.reset(train_mode=False)[
        brain_name]       # reset the environment
    # get the current state
    state = env_info.vector_observations[0]
    score = 0
    while True:
        action = agent.act(state)
        # send the action to the environment
        env_info = env.step(action)[brain_name]
        # get the next state
        next_state = env_info.vector_observations[0]
        reward = env_info.rewards[0]                        # get the reward
        score += reward
        # see if episode has finished
        done = env_info.local_done[0]
        # roll over the state to next time step
        state = next_state
        if done:
            print("Score: {}".format(score))
            break

env.close()
