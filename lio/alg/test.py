"""
This file is used to test whether the parameters of the policy network have been updated properly.
"""

import numpy as np
import random
import gym
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torchmeta.utils.gradient_based import gradient_update_parameters
from collections import OrderedDict
from matplotlib import pyplot as plt

from lio.agent.lio_agent_pg import Actor
from lio.model.actor_net import Trajectory
from lio.alg import config_room_lio
from lio.env import room_symmetric
from lio.model.meta_actor_net import (MetaNet_PG, weight_init_meta)
from lio.utils.util import Adam_Optim


# set random seed
seed = 3333
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)


def test_policy_update(config):
    agents = []
    for i in range(config.env.n_agents):
        agents.append(Actor(i, 7, config.env.n_agents))

    input = torch.Tensor([1, 1, 1, 1, 1, 1, 1])

    agent0 = agents[0]
    agent1 = agents[1]

    output0 = agent0.policy_net(input)
    output1 = agent1.policy_net(input)
    loss0 = 1 - output0.sum()
    loss1 = 2 - output1.sum()

    print(loss0)
    print(loss1)

    agent0.new_params = gradient_update_parameters(agent0.policy_net, loss0, step_size=0.5)
    agent1.new_params = gradient_update_parameters(agent1.policy_net, loss1, step_size=0.5)

    output0 = agent0.policy_net(input, agent0.new_params)
    output1 = agent1.policy_net(input, agent1.new_params)
    loss0 = 1 - output0.sum()
    loss1 = 2 - output1.sum()

    print(loss0)
    print(loss1)

    agent0.update_to_new_params()
    agent1.update_to_new_params()

    output0 = agent0.policy_net(input)
    output1 = agent1.policy_net(input)
    loss0 = 1 - output0.sum()
    loss1 = 2 - output1.sum()

    print(loss0)
    print(loss1)


def test_net_init():
    net = MetaNet_PG(7, 3, 64, 32)
    test_name = "actor_net.0.weight"
    for name, param in net.named_parameters():
        if name == test_name:
            print("name: ", name, "\nvalue: ", param)
    net.apply(weight_init_meta)
    print("----------------------------------------")
    for name, param in net.named_parameters():
        if name == test_name:
            print("name: ", name, "\nvalue: ", param)


def test_reward_net():
    agents = []
    for i in range(2):
        agents.append(Actor(i, 7, 2))

    inputs = np.arange(0, 10)
    inputs = torch.Tensor(inputs)
    outputs = agents[0].reward_net(inputs)
    print(outputs)

    inputs = np.arange(1, 11)
    inputs = torch.Tensor(inputs)
    outputs = agents[0].reward_net(inputs)
    print(outputs)


def test_pg():
    reward_list = []
    env = gym.make('CartPole-v0')
    agent = Actor(0, 4, 1, n_action=2)

    for epoch in range(5000):
        traj = [Trajectory()]
        list_obs = env.reset()
        list_obs_next = None
        done = False
        total_reward = 0

        while not done:
            if list_obs_next is not None:
                list_obs = list_obs_next

            inputs = torch.Tensor(list_obs)
            with torch.no_grad():
                probs = F.softmax(agent.policy_net(inputs), dim=0)
                m = Categorical(probs)
                action = m.sample()
                action = action.item()

            list_obs_next, reward, done, info = env.step(action)
            traj[0].add(list_obs, action, 0, reward, 0)
            total_reward += reward

            if done:
                reward_list.append(total_reward)

        agent.update_policy_op(traj)
        agent.update_to_new_params()

    env.close()
    plt.figure()
    plt.plot(reward_list)
    plt.show()


def test_adam():
    agent = Actor(0, 4, 1, n_action=2)

    for (name, param) in agent.policy_net.named_parameters():
        if name == 'actor_net.0.weight':
            print(name, ':', param)
    inputs = torch.Tensor([100, 200, 300, 400])
    outputs = agent.policy_net(inputs)
    outputs = torch.log(outputs.sum())
    print(outputs.item())

    agent.new_params = agent.optimizer_p.update(agent.policy_net, outputs)
    agent.update_to_new_params()

    for (name, param) in agent.policy_net.named_parameters():
        if name == 'actor_net.0.weight':
            print(name, ':', param)
    new_outputs = agent.policy_net(inputs)
    new_outputs = torch.log(new_outputs.sum())
    print(new_outputs.item())


def test_no_nan():
    eps = torch.finfo(torch.float32).eps

    x = torch.tensor([1, 2, 0], dtype=torch.float, requires_grad=True)

    y = torch.sqrt(x + eps)
    loss = torch.nn.functional.mse_loss(x, y)
    loss.backward()
    print(x.grad)


if __name__ == "__main__":
    config = config_room_lio.get_config()
    with torch.autograd.set_detect_anomaly(True):
        test_no_nan()
