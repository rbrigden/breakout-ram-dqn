import gym
import copy
import time
from util import *


model = os.environ["MODEL"]
dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
env = gym.make('Breakout-ram-v0')
q = DQN_RAM()
q.load_state_dict(torch.load(MODEL)["state_dict"])
episode_rewards = []
episode_reward = 0
episode_actions = np.zeros(6)
episode_actions.dtype = int
for e in range(10):
    print "Episode {} begun".format(e)
    print "actions taken: {}".format(episode_actions)
    print "episode_rewards: {}".format(episode_rewards)
    env.reset()
    x, r, terminate, _  = env.step(3)
    s = torch.from_numpy(x).type(dtype)
    episode_reward = 0
    episode_actions = np.zeros(6)
    episode_actions.dtype = int
    terminated = False
    while not terminated:
        env.render()
        qs = q(Variable(s).unsqueeze(0))
        action = argmax(qs.t().data)
        episode_actions[action] += 1
        nx, r, terminate, _  = env.step(3)
        ns = torch.from_numpy(nx).type(dtype)
        episode_reward += r
        s = ns
    episode_rewards.append(episode_reward)
