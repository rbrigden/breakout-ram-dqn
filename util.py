# util functions
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable
from hyperparams import *
import notify
import json

dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

class DQN_RAM(nn.Module):
    def __init__(self, in_features=128, num_actions=6):
        super(DQN_RAM, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, num_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.fc4(x)


class Experience(object):
    def __init__(self, s, a, r, ns, term):
        self.s = s
        self.a = a
        self.r = r
        self.ns = ns
        self.term = term


class Agent(object):

    def __init__(self):
        self.memory_size = REPLAY_MEMORY_SIZE
        self.memory = list()

    def __repr__(self):
        return "Agent {}".format(self.piece)

    def remember(self, s, a, r, ns, term):
        experience = Experience(s, a, r, ns, term)
        if len(self.memory) > self.memory_size:
            replace_idx = int(np.random.rand()*self.memory_size) % self.memory_size
            self.memory[replace_idx] = experience
        else:
            self.memory.append(experience)

    def sample(self,n):
        return np.random.choice(self.memory, n)

def argmax(x):
    _, i = torch.max(x, 0)
    return i[0][0]

def explore(eps):
    return np.random.rand() < eps


def empty_arr(n):
    x = np.zeros(n)
    x.dtype = int
    return x
