import gym
import copy
import time
from util import *
from datetime import datetime


# CUDA Support
USE_CUDA = torch.cuda.is_available()
if USE_CUDA:
    print "it's CUDA bitch"
dtype = torch.cuda.FloatTensor if USE_CUDA else torch.FloatTensor
ltype = torch.cuda.LongTensor if USE_CUDA else torch.LongTensor
btype = torch.cuda.ByteTensor if USE_CUDA else torch.ByteTensor



class Variable(autograd.Variable):

    def __init__(self, data, *args, **kwargs):
        if USE_CUDA:
            data = data.cuda()
        super(Variable, self).__init__(data, *args, **kwargs)

# MODELS
Q = DQN_RAM().cuda() if USE_CUDA else DQN_RAM()
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
episodes = 0
episode = 0
epoch = 0
estart = datetime.now()
while updates < EPOCH_SIZE * EPOCHS:
    print "\nEpisode {} begun".format(episode)
    episode += 1
    epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * \
    math.exp(-1. * actions_taken / EPSILON_DECAY)
    if updates % EPOCH_SIZE == 0:
        ed = str(datetime.now() - estart)
        estart = datetime.now()
    if updates % (NOTIFY_RATE * EPOCH_SIZE) == 0:
        with open('notify_receivers.json') as data_file:
            receivers = json.load(data_file)
        curr_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        subject = "Training Update Epoch {}".format(epoch)
        body = "Just updating you on the training process. The time is\
        {} and I will update you every {} epochs. {} more to go!\
        Here are some stats to let you know how well I am doing:".format(\
        curr_time, NOTIFY_RATE, EPOCHS - epoch)
        data = {"Mean reward":np.mean(episode_rewards),
                "Median reard":np.median(episode_rewards),
                "Raw rewards by episode":episode_rewards,
                "Last epoch duration":ed}

        notify.send(subject=subject,
                    data = data,
                    body = body,
                    to = receivers["emails"])
        epoch += 1
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
    print "Updates: {}".format(updates)
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
        agent.remember(s, action, r, ns, terminated)

        # sample from B
        if (frames_seen > BATCHSIZE):
            # print "Updating"
            samples = agent.sample(BATCHSIZE)

            # Compute a mask of non-final states and concatenate the batch elements
            non_final_mask = torch.Tensor(tuple(map(lambda e: e.term is False, samples))).type(torch.ByteTensor)

            # We don't want to backprop through the expected action values and volatile
            # will save us on temporarily changing the model parameters'
            # requires_grad to False!
            non_final_next_states = Variable(torch.stack([e.ns for e in samples
                                                        if e.term is False], 0),
                                             volatile=True)
            state_batch = Variable(torch.stack(tuple([e.s for e in samples]), 0))
            action_batch = Variable(torch.from_numpy(np.array([e.a for e in samples])).type(ltype))
            reward_batch = Variable(torch.from_numpy(np.array([e.r for e in samples])).type(dtype))
            state_action_values = Q(state_batch).gather(1, action_batch.unsqueeze(1))
            next_state_values = Variable(torch.zeros(BATCHSIZE)).cpu()
            next_state_values[non_final_mask] = Q(non_final_next_states).max(1)[0]
            next_state_values.volatile = False
            if USE_CUDA:
                expected_state_action_values = (next_state_values.cuda() * GAMMA) + reward_batch
            else:
                expected_state_action_values = (next_state_values * GAMMA) + reward_batch


            loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

            optimizer.zero_grad()
            loss.backward()
            for param in Q.parameters():
                param.grad.data.clamp_(-1, 1)
            optimizer.step()
            updates += 1

        s = ns
        frames_seen += 1
    episode_rewards.append(episode_reward)


torch.save({
            'state_dict': Q.state_dict(),
            }, MODEL_SAVE_PATH )
