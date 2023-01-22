# Udacity Deep Reinforcement Learning Project 1: Navigation

### Project Files

- **README.md** (this file)

- **Report.ipynb**: *the completed project notebook*. All work on this project is located in this file.

- *Navigation.ipynb*: starter code and scratchpad notebook, **NOT** the completed project.

- The **dqn_agent.py** and **model.py** files in the project folder have been copied as-is from the [DQN-Solution folder](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution).

- The **dqn_solution.py** file contains a very slightly modified version of the function provided as part of the course in the [Deep Q-Network Solution notebook](https://github.com/udacity/deep-reinforcement-learning/tree/master/dqn/solution/dqn.ipynb).

- **benchmark_weights.pth**: stored weights of an Agent's local QNetwork, saved right after achieving the benchmark average score of 13+
- **trained_weights.pth**:  stored weights after training on a full set of 2000 episodes, achieving a final average score of about 16.

#### **References and Resources**

- Background and design of Deep Q-Learning models and training methodologies covered in the [Course Textbook](http://go.udacity.com/rl-textbook) Chapter 6 Section 5.
- DQN algorithms used in the project developed from [this paper in *Nature*](https://www.nature.com/articles/nature14236) 
 - Mnih, V., Kavukcuoglu, K., Silver, D. *et al.* Human-level control through deep reinforcement learning. *Nature* **518**, 529–533 (2015). https://doi.org/10.1038/nature14236

- [Unity Machine Learning Agents](https://blogs.unity3d.com/2017/09/19/introducing-unity-machine-learning-agents/) Environment Resource

### Project Description and UnityAgents Environment

A reinforcement learning agent will be trained to navigate on a 2D surface by choosing in which direction to travel given 37 dimensions of information about its environment at each time step, such as its location, momentum, and the distance from it to the bananas and walls in its "view". The agent must respond to each state with one of these four actions:

- **`0`** - move forward.
- **`1`** - move backward.
- **`2`** - turn left.
- **`3`** - turn right.

The goal of training is to develop a policy of actions maximizing the agent's collisions with yellow bananas while minimizing its collisions with blue bananas. Therefore at each step the agent recieves one of three awards according to the outcome of the agent's choice of action:
-  **` 0`**  - no collision, or collision with wall
-  **`+1`** - successfully running over a yellow banana
-  **`-1`** - any contact with a blue banana

Episodes end only after a pre-specified number of steps (with the default apparently set to a maximum of 300 steps by the environment). An episode's final score is the unweighted sum of its rewards after each step.

The agent will use Deep Q-Learning techniques to asymptotically approximate the optimal policy for collecting as many yellow bananas as possible while avoiding blue bananas over the preset number of steps for each episode.  

The project rubric requires the agent to achieve an average score of at least 13 (over 100 consecutive episodes) after 1800 training episodes, or less.
