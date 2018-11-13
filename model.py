import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.in_fc1 = nn.Linear(state_size,32)
        self.hidden = nn.Linear(32,16)
        #self.hidden2 = nn.Linear(128,64)
        #self.hidden3 = nn.Linear(64,32)
        self.output = nn.Linear(16,action_size)
        
        #dropout layer
        #self.dropout = nn.Dropout(0.4)

    def forward(self, state):
        """Build a network that maps state -> action values."""
        
        state = F.relu(self.in_fc1(state))
        state = F.relu(self.hidden(state))
        #state = self.dropout(state)
        #state = F.relu(self.hidden2(state))
        #state = F.relu(self.hidden3(state))
        state = self.output(state)
        
        return state

class Duel_QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(Duel_QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.in_fc1 = nn.Linear(state_size, 64)
        self.hidden = nn.Linear(64, 64)
        #self.hidden2 = nn.Linear(256, 64)
        self.value = nn.Linear(64, action_size)
        self.advantage = nn.Linear(64, action_size)
        #self.output = nn.Linear(32, action_size)

        # dropout layer
        self.dropout = nn.Dropout(0.4)

    def forward(self, state):
        """Build a network that maps state -> action values."""

        state = F.relu(self.in_fc1(state))
        state = F.relu(self.hidden(state))
        #state = F.relu(self.hidden2(state))
        #state = self.dropout(state)

        value = self.value(state)
        advantage = self.advantage(state)
        state = value + (advantage - torch.mean(advantage))

        return state