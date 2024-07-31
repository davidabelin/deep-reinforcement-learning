import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=256, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)

    def forward(self, state):
        """Assign action_values to states according to current policy
           Params 
           state (list-like) of values from environent of shape (batch_size, state_size)
           Returns: a Q-value estimate for each possible action: "OFF", "MAIN", "RIGHT", "LEFT"
           These values can be converted into relative probabilities by passing the values through a
           softmax function. An epsilon-greedy choice can be made on either to determine the action taken.
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class QQNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, seed, fc1_units=64, fc2_units=32):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QQNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.state0 = nn.Linear(state_size, state_size)
        self.state1 = nn.Linear(state_size, state_size)
        self.hidden = nn.Linear(2*state_size, 8*action_size)
        self.out = nn.Linear(8*action_size, action_size)

    def forward(self, state):
        """Build a network that maps state -> action values.
           Shape of input state should be: (batch_size, 2, state_size)
        """
        s0 = state[:,0,:].squeeze()
        s0 = F.relu(self.state0(s0))
        s1 = state[:,1,:].squeeze()
        s1 = F.relu(self.state1(s1))                       
        x = torch.cat((s0,s1), dim=-1)
        x = F.relu(self.hidden(x))
        return self.out(x)