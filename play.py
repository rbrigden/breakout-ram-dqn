import gym
import copy
import time
from util import *


MODEL = "./mymodel.mdl"

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dtype = torch.FloatTensor
env = gym.make('Breakout-v0')
q = DQN_RAM()
q.load_state_dict(torch.load(MODEL)["state_dict"])
episode_rewards = []
episode_reward = 0
episode_actions = np.zeros(6)
episode_actions.dtype = int
for e in range(10):
    print "Episode {} begun".format(e)
    x = env.reset()
    x = torch.from_numpy(x).type(dtype)
    s, r, terminate, _  = env.step(3)
    print "actions taken: {}".format(episode_actions)
    print "episode_rewards: {}".format(episode_rewards)
    episode_reward = 0
    episode_actions = np.zeros(6)
    episode_actions.dtype = int
    terminated = False
    while not terminated:
        env.render()
        qs = q(s)
        action = argmax(qs.t().data)
        episode_actions[action] += 1
        nx, r, terminate, _  = env.step(3)
        ns = torch.from_numpy(nx).type(dtype)
        episode_reward += r
        s = ns
    episode_rewards.append(episode_reward)
