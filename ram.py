import gym
import copy
import time
from util import *

# CUDA Support
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print "it's CUDA bitch"
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
ltype = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor


class Variable(autograd.Variable):

    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

# MODELS
Q = DQN_RAM().cuda() if USE_CUDA else DQN_RAM()
Q_hat = copy.deepcopy(Q)
optimizer = torch.optim.RMSprop(Q.parameters())
updates = 0

env = gym.make('Breakout-ram-v0')
seq = []
agent = Agent()
actions_taken = 0
frames_seen = 0
episode_actions = empty_arr(6)
explored_actions = empty_arr(6)
episode_reward = 0
episode_rewards = []
for e in range(EPISODES):
    print "\nEpisode {} begun".format(e)
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
    math.exp(-1. * actions_taken / EPSILON_DECAY)
    # first frame
    x = env.reset()
    s = torch.from_numpy(x).type(dtype)
    # episode metrics
    print "actions taken by net: {}".format(episode_actions - explored_actions)
    print "episode_rewards: {}".format(episode_rewards)
    print "Epsilon: {}".format(epsilon)
    episode_reward = 0
    episode_actions = empty_arr(6)
    explored_actions = empty_arr(6)
    # start inner loop
    terminated = False
    while not terminated:

        env.render()

        if explore(epsilon):
            action = env.action_space.sample()
            explored_actions[action] += 1
        else:
            qs = Q(Variable(s.view(1, 128)))
            actions_taken += 1
            action = argmax(qs.t().data)

        episode_actions[action] += 1
        x_p, r, terminated, _ = env.step(action)
        ns = torch.from_numpy(x_p).type(dtype)
        episode_reward += r

        # we only add the q approx to the reward for terminal states
        # mask must be 0 in the terminal state case only
        agent.remember(s, action, r, ns, int(not terminated))

        # sample from B
        if (frames_seen > BATCHSIZE):
            # print "Updating"
            samples = agent.sample(BATCHSIZE)
            next_states = Variable(torch.cat([e.ns.view(1, 128) for e in samples]))
            curr_states = Variable(torch.cat([e.s.view(1, 128) for e in samples]))
            curr_rewards = np.array([ e.r for e in samples ])
            curr_actions = np.array([ e.a for e in samples ])

            term_mask = np.array([e.term for e in samples])
            curr_rewards = Variable(torch.from_numpy(curr_rewards).type(dtype))
            curr_actions = Variable(torch.from_numpy(curr_actions).type(ltype))
            term_mask = Variable(torch.from_numpy(term_mask).type(dtype))

            if USE_CUDA:
                curr_actions = curr_actions.cuda()
                curr_rewards = curr_rewards.cuda()
                term_mask = term_mask.cuda()

            curr_state_qs = Q(curr_states).gather(1, curr_actions.unsqueeze(1))
            next_state_qs = Q_hat(next_states).detach().max(1)[0]
            targets = torch.add(curr_rewards, (GAMMA * (term_mask * next_state_qs)))
            # err = (targets - curr_state_qs) # MSELoss
            loss = F.smooth_l1_loss(curr_state_qs, targets)
            # print loss
            # Forward + Backward + Optimize
            optimizer.zero_grad()
            loss.backward()
            for param in Q.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            updates += 1


            if (updates % SWAPRATE == 0):
                print "swap!"
                Q_hat.load_state_dict(Q.state_dict())

        s = ns
        frames_seen += 1
    episode_rewards.append(episode_reward)


torch.save({
            'state_dict': Q.state_dict(),
            }, MODEL_SAVE_PATH )
