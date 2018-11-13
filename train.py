from unityagents import UnityEnvironment
import random
import torch
import numpy as np
from collections import deque
import config

from datetime import datetime

import matplotlib.pyplot as plt

is_windows = True

# Note I had no luck running it on Linux - there is some fault in Unity and doesn't load plugins correctly
env = UnityEnvironment(file_name="./Banana_Windows_x86_64/Banana.exe" if is_windows else "./Banana_Linux/Banana.x86_64")

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

print(brain_name)

# reset the environment
env_info_x = env.reset(train_mode=True)[brain_name]

# number of agents in the environment
print('Number of agents:', len(env_info_x.agents))

# number of actions
action_size = brain.vector_action_space_size
print('Number of actions:', action_size)

# examine the state space
state = env_info_x.vector_observations[0]
print('States look like:', state)
state_size = len(state)
print('States have length:', state_size)

#Define agent algorithm

#from doubledqn_agent import Agent
from duel_dqn_agent_PER import Agent

#from duel_dqn_agent import Agent

agent = Agent(state_size=state_size, action_size=action_size, seed=random.randint(5,100), train= (not config.PLAY_ONLY))

from time import sleep


def extract_env_details(env_info):
    next_state = env_info.vector_observations[0]  # get the next state
    reward = env_info.rewards[0] # get the reward
    done = env_info.local_done[0]  # see if episode has finished

    return next_state, reward, done

actions = []

# 400 with g=0.1 worked up to 18
def dqn(train_agent = True, n_episodes=1000, max_t=1000, eps_start=1.0, eps_end=0.001, eps_decay=0.975):
    """Deep Q-Learning.

    Params
    ======
        train_agent - should the agent be trained or use predefined weights
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []  # list containing scores from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start  # initialize epsilon

    # decayng weights over time helps with training stability
    scheduler = torch.optim.lr_scheduler.StepLR(agent.optimizer, step_size=200, gamma=0.20)

    # set threshold for saving high-scores - as this model almost always get's to the region of 16-17
    avg_max_score = 16.0
    for i_episode in range(1, n_episodes + 1):
        # decay weights
        scheduler.step()
        env_info = env.reset(train_mode=train_agent)[brain_name]

        #print(f"Episode {i_episode} Env info {env_info}")
        state = env_info.vector_observations[0]
        score = 0

        total_steps = 0
        while True:

            action = agent.act(state, eps)

            total_steps += 1

            env_info = env.step(int(action))[brain_name]
            next_state, reward, done = extract_env_details(env_info)
            if train_agent:
                agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward

            if done:
                break

        # Episode is over ; updating stats
        scores.append(score)  # save most recent score
        scores_window.append(score)  # save most recent score

        eps = max(eps_end, eps_decay * eps)  # decrease epsilon

        if not train_agent and (i_episode % 100 == 0):
            avg_score = np.mean(scores_window)
            print('\nEpisode {}\tAverage Score: {:.2f} Max score: {}'.format(i_episode, avg_score, np.max(scores)), end=" ")
            break

        # print every 5th episode but don't duplicate 100
        if (i_episode % 5 == 0) & (i_episode % 100 != 0):
            avg_score = np.mean(scores_window)
            last_max_score = avg_max_score
            avg_max_score = max(avg_max_score,avg_score)
            print('\nEpisode {}\tAverage Score: {:.2f} Max score: {}'.format(i_episode, avg_score, np.max(scores)), end=" ")

            # save max score in case it drops later
            if avg_max_score > last_max_score:
                print('\nNew Max score: {:.2f} in ep: {}'.format(avg_max_score, i_episode))
                torch.save(agent.qnetwork_local.state_dict(), "Max_score_Local_DualDQN_ep" + str(i_episode) + '.pth')

        if i_episode % 100 == 0:
            print('\nEpisode {}\tAverage Score: {:.2f}'.format(i_episode, avg_score))
            torch.save(agent.qnetwork_local.state_dict(), "DualDQN_ep" + str(i_episode) + '.pth')

        # In my experiments this network is capable of reaching level of 18pts so that's where i set the bar
        if np.mean(scores_window) >= 18.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(i_episode - 100,
                                                                                         np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), 'Over_18_' + str(i_episode) + '.pth')
            break


    return scores

def load_weights():
    pre_trained_weights = torch.load(config.SAVED_WEIGHTS_FILE)
    agent.qnetwork_local.load_state_dict(pre_trained_weights)

if config.PLAY_ONLY:
    print('Agent will use pretrained model')
    load_weights()
    scores = dqn(train_agent=False)

else:
    # trains the network and can take a while!
    scores = dqn()

# plot the scores
fig = plt.figure(figsize=(15,30))

ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()