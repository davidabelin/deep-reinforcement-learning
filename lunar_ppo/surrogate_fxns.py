import torch
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np

import lunar_PPO_utils as lpu
from lunar_PPO_utils import *

### "same thing as -policy_loss, but clipped"
def clipped_surrogate(policy, old_probs, 
                      states, actions, rewards,
                      discount=0.995,
                      epsilon=0.1, 
                      beta=0.01):
    '''clipped surrogate function
       params:
           policy (training network)
           old_probs (array): , from prior action-value estimates made from states
           states (array): un-normed states, shape: (batch, 8)
           gamma (float): discount factor, should be $\left(1-\frac{1}{N}\right)$, N=???
       returns:
           sum of log-prob divided by T clipped to (-1,1)
    
    '''  
    old_probs = np.asarray(old_probs, dtype=np.float32)
    actions = np.asarray(actions, dtype=np.int8)
    states = np.asarray(states, dtype=np.float32)
    
    ##### Pre-Norm States
    states = lpu.scale_input_batch( states.reshape(-1, 8) ).reshape(*states.shape)
    
    #### convert rewards to normed future rewards
    #gamma = min(discount, 1-(1/len(rewards)) if len(rewards)>0. else 1.)
    gamma = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*gamma[:,np.newaxis]
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-8
    rewards_normalized = (rewards_future - mean[:, np.newaxis])/std[:, np.newaxis]
    #############################################
     
    # convert everything into pytorch tensors and move to gpu if available
    states = torch.tensor(states, dtype=torch.float32, device=lpu.device)
    actions = torch.tensor(actions, dtype=torch.int8, device=lpu.device)
    old_probs = torch.tensor(old_probs, dtype=torch.float32, device=lpu.device)
    rewards = torch.tensor(rewards_normalized[:, :, np.newaxis], dtype=torch.float32, device=lpu.device)

    # convert states to policy (or probability)
    new_probs = lpu.states_to_probs(policy, states)
    
    # ratio for clipping
    ratio = new_probs/old_probs

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)

    # include a regularization term to steer new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -1*( new_probs*torch.log(old_probs+1.e-10) + (1.0-new_probs)*torch.log(1.0-old_probs+1.e-10) )

    
    # returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T averaged over time-step and number of trajectories
    # and  rewards have been normalized
    return torch.mean(clipped_surrogate + beta*entropy)


#####################################
# return sum of log-prob divided by T
# same thing as -policy_loss
def surrogate(policy, old_probs, states, actions, rewards, 
              discount = 0.995, beta=0.01):

    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    
    old_probs = np.asarray(old_probs)
    states = np.asarray(states)
    actions = np.asarray(actions)
    
    # convert everything into pytorch tensors and move to gpu if available
    states = torch.tensor(states, dtype=torch.int8, device=lpu.device)
    actions = torch.tensor(actions, dtype=torch.int8, device=lpu.device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=lpu.device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=lpu.device)

    # convert states to policy (or probability of actions)
    new_probs = lpu.states_to_probs(policy, states)

    ratio = new_probs/old_probs

    # include a regularization term to steer new_policy towards 0.5
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy =  -(new_probs*torch.log(old_probs+1.e-8)+(1.0-new_probs)*torch.log(1.0-old_probs+1.e-8))

    return torch.mean(ratio*rewards + beta*entropy)

    
# 
def clipped_surrogate_beta(policy, old_probs, states, actions, rewards, discount = 0.995, epsilon=0.1, beta=0.01):
    ## WRITE YOUR OWN CODE HERE  ## from plot_utils
    #####################?
    discount = discount**np.arange(len(rewards))
    rewards = np.asarray(rewards)*discount[:,np.newaxis]
    
    # convert rewards to future rewards
    rewards_future = rewards[::-1].cumsum(axis=0)[::-1]
    
    mean = np.mean(rewards_future, axis=1)
    std = np.std(rewards_future, axis=1) + 1.0e-10

    rewards_normalized = (rewards_future - mean[:,np.newaxis])/std[:,np.newaxis]
    ######################?
    
    # move to gpu if available
    actions = torch.tensor(actions, dtype=torch.int8, device=lpu.device)
    old_probs = torch.tensor(old_probs, dtype=torch.float, device=lpu.device)
    rewards = torch.tensor(rewards_normalized, dtype=torch.float, device=lpu.device)

    # convert states to policy (or probability)
    new_probs = pong_utils.states_to_prob(policy, states)
    new_probs = torch.where(actions == pong_utils.RIGHT, new_probs, 1.0-new_probs)

    # ratio for clipping
    ratio = new_probs/old_probs

    # clipped function
    clip = torch.clamp(ratio, 1-epsilon, 1+epsilon)
    clipped_surrogate = torch.min(ratio*rewards, clip*rewards)
    
    # include a regularization term
    # this steers new_policy towards 0.5
    # prevents policy to become exactly 0 or 1 helps exploration
    # add in 1.e-10 to avoid log(0) which gives nan
    entropy = -(new_probs*torch.log(old_probs+1.e-10)+(1.0-new_probs)*torch.log(1.0-old_probs+1.e-10))

    # this returns an average of all the entries of the tensor
    # effective computing L_sur^clip / T
    # averaged over time-step and number of trajectories
    # this is desirable because we have normalized our rewards
    return torch.mean(clipped_surrogate + beta*entropy)