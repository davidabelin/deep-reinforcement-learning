# taken from openai/baseline
# with minor edits
# see https://github.com/openai/baselines/baselines/common/vec_env/subproc_vec_env.py
# 

################# TO DO convert everything to work with Unity envs 

import gym
from multiprocessing import Process, Pipe
from abc import ABC, abstractmethod

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)

class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        """
        Step the environments synchronously.
        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        #logger.warn('Render not defined for %s' % self)
        pass
        
    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self


def worker(remote, parent_remote, env_fn_wrapper):
    ''' Mediates interaction between Pipes and Environments.
        remote (Pipe): parallelEnv.work_remote
        parent_remote (Pipe): parallelEnv.remote
        env_fn_wrapper: wraps a single gym env
    
    '''
    parent_remote.close()   #
    env = env_fn_wrapper.x  # return one unwrapped Env
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

################# TO DO convert to use with Unity envs 
#################
class parallelEnv(VecEnv):
    def __init__(self, env_name=None, n=4, seed=None, spaces=None):
        
        """
        VecEnv: the abstract class to instantiate with Gym Envs 
        env_name: kind of gym env to use for env_fns
        env_fns: "list of gym environments to run in subprocesses adopted from openai baseline"
        
        env_info = env.reset(train_mode=True)[brain_name]     # reset the environment, mode=False for display     
        states = env_info.vector_observations                  # get the current state (for each agent)
        scores = np.zeros(num_agents)                          # initialize the score (for each agent)
        actions = np.random.randn(num_agents, action_size)    # select an action (for each agent, (20, 4)
        actions = np.clip(actions, -1, 1)                    # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]             # send all actions to tne environment
        next_states = env_info.vector_observations           # get next state (for each agent)
        rewards = env_info.rewards                           # get reward (for each agent)
        dones = env_info.local_done                          # see if episode finished
        scores += env_info.rewards                           # update the score (for each agent)
        states = next_states                                 # roll over states to next time step
        if np.any(dones): break                              # exit loop if episode finished
        """

        #env_fns = [ gym.make(env_name) for _ in range(n) ]
        #if seed is not None:
        #    for i,e in enumerate(env_fns):
        #        e.seed(i+seed)    # only matters if it gets to where it matters
                                   ##################### TO DO: where?

        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        
        # Pipes
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        
        # Processes (workers, envs, env_wrappers)
        # 'work_remotes' become workers who relay game info between agent and env
        # 'remotes' are the parent_remotes of these work_remotes; closed immediately by worker init
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                  for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None)) 
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

####################################################
############# Not in Service #######################
####################################################
        
        
def retired_worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, trun, info = env.step(data)
            if done or trun:
                ob, info = env.reset()
            remote.send(ob, reward, done, trun, info)
        elif cmd == 'reset':
            ob, info = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            print("******* Worker closed *******")
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        else:
            raise NotImplementedError

class parallelDimension(VecEnv):        
    """
    Formerly known as  parallelEnv of lunar_PPO
    envs: list of gym environments to run in subprocesses
    adopted from openai baseline
    """
    def __init__(self,  env_name="LunarLander-v2",
                        n=4,
                        seed=1234,
                        spaces=None):
        
        self.envs = [ gym.make(env_name, render_mode='rgb_array') for _ in range(n) ]
        self.seeds = [ max(2*seed-i**2, i*(1+1/(seed+i+1e-6))) for i in range(n) ]
        for e, s in zip(self.envs, self.seeds):
            # each env has its own random seed
            _, _ = e.reset(seed=s)

        self.waiting = False
        self.closed = False
        self.n = n
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n)])
        self.ps = [Process(target=worker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, self.envs)]
        for p in self.ps:
            p.daemon = True # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, n, observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        states, rews, dones, truns, infos = zip(*results)
        return np.stack(states), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        print("Closing envs.....")
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:            
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True