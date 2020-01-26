import random
import numpy as np
import os
import torch
from .train import transform_state


class Agent:
    def __init__(self):
        self.Q = torch.nn.Linear(7, 3)
        with torch.no_grad():
            self.Q.weight[:], self.Q.bias[:] = np.load(__file__[:-8] + "/agent.npz")

    def act(self, state):
        return np.argmax(self.Q(transform_state(state)).detach().numpy())

    def reset(self):
        pass

