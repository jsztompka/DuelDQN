import numpy as np
import random
from collections import namedtuple, deque

# Import Dual network model for the agent
from model import Duel_QNetwork

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from custom_loss import huber_loss


from per import PrioritisedExpReplay

#All parameters are stored in the config (LR/Buffer size / etc.)
import config

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed,train = True):
        """Initialize an Agent object.
        
        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
            train - is agent being trained
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)
        self.train = train

        # Duel Double Q-Network
        self.qnetwork_local = Duel_QNetwork(state_size, action_size, seed).to(config.device)
        self.qnetwork_target = Duel_QNetwork(state_size, action_size, seed).to(config.device)

        # Adam was showing best results in my tests , adding amsgrad=True appeared to increase the learning slightly
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=config.LR, amsgrad=True)

        # Prioritised Experience Replay
        self.memory = PrioritisedExpReplay(config.BUFFER_SIZE, config.BATCH_SIZE, seed)

        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

        self.target_net_step = 0

        print(f"Using Agent defined in {type(self)} with LR={config.LR} and update rate={config.UPDATE_EVERY}")
    
    def step(self, state, action, reward, next_state, done):
        # Save experience in PER (wrapped in a tuple so that it can sent as a single argument)
        self.memory.store((state, action, reward, next_state, done))

        # This effectively skips learning step on UPDATE_EVERY steps
        self.t_step = (self.t_step + 1) % config.UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > config.BATCH_SIZE:
                # PER needs to be updated with Weights and tree_indexes (specific to this implementation)
                # we extract them all here and pass to learn where they will be applied
                tree_idx, experiences, IS_weights = self.memory.sample()
                self.learn(experiences, config.GAMMA, tree_idx, IS_weights)

    def act(self, state, eps=0.):
        """Returns actions for given state as per current policy.
        
        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(config.device)
        
        #sets network in eval mode which means no gradient is calculated (yet)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection or if it's not training
        if random.random() > eps or (not self.train):
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma, tree_idx, IS_weights):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples 
            gamma (float): discount factor
            tree_idx - indexes which will be used to update SumTree (PER)
            IS_weights - weights to apply to the loss function - (part of PER requirement)
        """
        states, actions, rewards, next_states, dones = experiences


        self.qnetwork_local.eval()
        self.qnetwork_target.eval()
        

        #returns argmax along axis=1 and then wraps the tensor in another tensor pushing it's rank + 1
        local_actions = self.qnetwork_local(next_states).max(1)[1].unsqueeze(1)


        target_action_values = self.qnetwork_target(next_states).gather(1,local_actions)
        action_values_current = self.qnetwork_local(states).gather(1,actions)

        # Calculate expected return based on Target network (Double network specific)
        expected = rewards + (gamma * target_action_values * (1 - dones)) 

        # PER needs to be updated with absolute errors in order to calculate relevant priorities
        absolute_errors = torch.abs(action_values_current - expected)
        self.memory.batch_update(tree_idx, absolute_errors.detach().cpu().numpy())

        # zero the parameter (weight) gradients
        self.optimizer.zero_grad()

        self.qnetwork_local.train()
        self.qnetwork_target.train()

        # Original DQN paper mentioned they were using Huber-Loss and while PyTorch has it's own implementation
        # that implementation doesn't allow for custom weights (part of PER requirement)
        # as a result i had to reimplement huber-loss (this one comes from custom_loss module)
        loss = huber_loss(action_values_current, expected, torch.as_tensor(IS_weights).to(config.device))

        # backward pass to calculate the parameter gradients
        loss.backward()

        # update the parameters
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, config.TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter 
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



