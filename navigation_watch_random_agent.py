from unityagents import UnityEnvironment
import numpy as np

# 1. Create environment
env = UnityEnvironment(file_name="simulator/Banana.x86_64")
brain_name = env.brain_names[0]  # get default brain
brain = env.brains[brain_name]
env_info = env.reset(train_mode=False)[brain_name]  # reset environment
action_size = brain.vector_action_space_size  # number of actions
state = env_info.vector_observations[0]  # get state space
state_size = len(state)

# 2. Perform random actions
score = 0                                          # initialize the score
while True:
    action = np.random.randint(action_size)        # select an action
    # send the action to the environment
    env_info = env.step(action)[brain_name]
    next_state = env_info.vector_observations[0]   # get the next state
    reward = env_info.rewards[0]                   # get the reward
    # see if episode has finished
    done = env_info.local_done[0]
    score += reward                                # update the score
    # roll over the state to next time step
    state = next_state
    if done:                                       # exit loop if episode finished
        break

print("Score: {}".format(score))

env.close()
