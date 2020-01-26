from gym import make
import numpy as np
import torch
import copy
from collections import deque
import random
from tqdm import tqdm

N_STEP = 3
GAMMA = 0.96


def transform_state(state):
    state = np.array(state)
    state = np.concatenate(((np.sign(state[1]),), state + np.array((1.2, 0.0)) / np.array((1.8, 0.07)), np.sin(state), np.cos(state)))
    result = []
    result.extend(state)
    return np.array(result)


class AQL:
    def __init__(self, state_dim, action_dim):
        self.gamma = GAMMA ** N_STEP
        self.Q = torch.nn.Linear(7, 3)
        with torch.no_grad():
            self.Q.weight[:] = torch.zeros((3, 7))
            self.Q.weight[0, 0] = -1
            self.Q.weight[2, 0] = 1
            self.Q.bias[:] = 0

    def act(self, state, target=False):
        return np.argmax(self.Q(torch.FloatTensor(state)).detach().numpy())

    def update(self, transition):
        state, action, next_state, reward, done = transition
        L2 = 0.999
        alpha = 0.0001
        *d_weight, d_bias = alpha * torch.FloatTensor(np.concatenate((state, (1,)))) * (self.Q(torch.FloatTensor(next_state))[action] - reward)
        with torch.no_grad():
            self.Q.weight[action, :] += torch.FloatTensor(d_weight)
            self.Q.weight *= L2
            self.Q.bias[action] += d_bias
            self.Q.bias *= L2

    def save(self, path):
        with torch.no_grad():
            np.savez("agent.npz", self.Q.weight.data.numpy(), self.Q.bias.data.numpy())


if __name__ == "__main__":
    env = make("MountainCar-v0")
    aql = AQL(state_dim=2, action_dim=3)
    eps = 0.1
    episodes = 2000

    for i in tqdm(range(episodes)):
        state = transform_state(env.reset())
        total_reward = 0
        steps = 0
        done = False
        reward_buffer = deque()
        state_buffer = deque()
        action_buffer = deque()
        while not done:
            if random.random() < eps:
                action = env.action_space.sample()
            else:
                action = aql.act(state)
            next_state, reward, done, _ = env.step(action)
            next_state = transform_state(next_state)
            total_reward += reward
            steps += 1
            reward_buffer.append(reward)
            state_buffer.append(state)
            action_buffer.append(action)
            if len(reward_buffer) == N_STEP:
                aql.update((state_buffer[0], action_buffer[0], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(reward_buffer)]), done))
            state = next_state
        if len(reward_buffer) == N_STEP:
            rb = list(reward_buffer)
            for k in range(1, N_STEP):
                aql.update((state_buffer[k], action_buffer[k], next_state, sum([(GAMMA ** i) * r for i, r in enumerate(rb[k:])]), done))
        if i % 20 == 0:
            aql.save('foo')
