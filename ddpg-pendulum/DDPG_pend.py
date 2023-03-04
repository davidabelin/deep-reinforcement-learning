#!/usr/bin/env python
# coding: utf-8

# # Deep Deterministic Policy Gradients (DDPG)
# ---
# In this notebook, we train DDPG with OpenAI Gym's Pendulum-v0 environment.
# 
# ### 1. Import the Necessary Packages

# In[1]:


import gym
import random
import torch
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
#%matplotlib inline

from ddpg_agent_pend import Agent


# ### 2. Instantiate the Environment and Agent

# In[2]:


env = gym.make('Pendulum-v0')
seed=2
#env.seed(2)
agent = Agent(state_size=3, action_size=1, random_seed=seed)


# ### 3. Train the Agent with DDPG

# In[3]:


def ddpg(n_episodes=1000, max_t=300, print_every=100):
    scores_deque = deque(maxlen=print_every)
    scores = []
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        agent.reset()
        score = 0
        for t in range(max_t):
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            agent.step(state, action, reward, next_state, done)
            state = next_state
            score += reward
            if done:
                break 
        scores_deque.append(score)
        scores.append(score)
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)), end="")
        torch.save(agent.actor_local.state_dict(), 'checkpoint_actor.pth')
        torch.save(agent.critic_local.state_dict(), 'checkpoint_critic.pth')
        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_deque)))
            
    return scores


# In[ ]:



scores = ddpg(n_episodes=30, max_t=50, print_every=10)

fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores)+1), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()


# ### 4. Watch a Smart Agent!

# In[ ]:


#agent.actor_local.load_state_dict(torch.load('checkpoint_actor_pend.pth'))
#agent.critic_local.load_state_dict(torch.load('checkpoint_critic_pend.pth'))

state = env.reset(seed=seed)
for t in range(200):
    action = agent.act(state, add_noise=False)
    env.render()
    state, reward, done, _ = env.step(action)
    if done:
        break 


# In[ ]:



#env.close()


# ### 6. Explore
# 
# In this exercise, we have provided a sample DDPG agent and demonstrated how to use it to solve an OpenAI Gym environment.  To continue your learning, you are encouraged to complete any (or all!) of the following tasks:
# - Amend the various hyperparameters and network architecture to see if you can get your agent to solve the environment faster than this benchmark implementation.  Once you build intuition for the hyperparameters that work well with this environment, try solving a different OpenAI Gym task!
# - Write your own DDPG implementation.  Use this code as reference only when needed -- try as much as you can to write your own algorithm from scratch.
# - You may also like to implement prioritized experience replay, to see if it speeds learning.  
# - The current implementation adds Ornsetein-Uhlenbeck noise to the action space.  However, it has [been shown](https://blog.openai.com/better-exploration-with-parameter-noise/) that adding noise to the parameters of the neural network policy can improve performance.  Make this change to the code, to verify it for yourself!
# - Write a blog post explaining the intuition behind the DDPG algorithm and demonstrating how to use it to solve an RL environment of your choosing.  
