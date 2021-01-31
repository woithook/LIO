"""
Reproduction of LIO in the Escape Room

Version 0.0:
Constructed a whole framework consisting of policy network with
A2C and reward giving network. policy net works but reward giving
net can not learning correctly.

ToDo:
To fix reward giving net by checking the gradient graph.
To rearrange the duplicated code with unified method.
"""

import numpy as np
import random
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt

from lio.agent.lio_agent import Actor
from lio.model.actor_net import Trajectory

from lio.alg import config_room_lio
from lio.env import room_symmetric


def train(config):

    # set seeds for the training
    # seed = config.main.seed
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period

    results_one = []
    results_two = []
    reward_one_to_two = []
    reward_two_to_one = []

    # 初始化环境
    env = room_symmetric.Env(config.env)

    agents = []
    for i in range(config.env.n_agents):
        agents.append(Actor(i, 7, config.env.n_agents))

    # epoch start
    for epoch in range(5000):
        trajs = [Trajectory() for _ in range(env.n_agents)]
        list_obs = env.reset()
        list_obs_next = None
        done = False
        result_one = 0
        result_two = 0

        while not done:
            list_act = []
            list_act_hot = []
            if list_obs_next is not None:
                list_obs = list_obs_next
            # set observations and decide actions
            for agent in agents:
                agent.set_obs(list_obs[agent.id])
                agent.action_sampling()
                list_act_hot.append(agent.get_action_hot())
                list_act.append(agent.get_action())

            list_rewards = []
            total_reward_given_to_each_agent = torch.zeros(env.n_agents)

            # give rewards
            for agent in agents:
                reward = agent.give_reward(list_act_hot)
                for idx in range(env.n_agents):
                    if idx != agent.id:
                        total_reward_given_to_each_agent[idx] += reward[idx]
                reward_sum = reward.sum().detach().numpy()  # 计算自己总共给予了多少报酬
                list_rewards.append(reward_sum)

            # execute step
            list_obs_next, env_rewards, done = env.step(list_act, list_rewards)

            for agent in agents:
                reward_given = total_reward_given_to_each_agent[agent.id]
                trajs[agent.id].add(agent.get_obs(), agent.get_action(), agent.get_action_hot(), env_rewards[agent.id], reward_given)

            result_one += env_rewards[0]
            result_two += env_rewards[1]

            # if done:
                # results_one.append(result_one)
                # results_two.append(result_two)

        for agent in agents:
            agent.update_policy(trajs[agent.id])

        # Generate a new trajectory
        trajs_new = [Trajectory() for _ in range(env.n_agents)]
        list_obs = env.reset()
        list_obs_next = None
        done = False
        result_one_new = 0
        result_two_new = 0

        while not done:
            list_act = []
            list_act_hot = []
            if list_obs_next is not None:
                list_obs = list_obs_next
            # set observations and decide actions
            for agent in agents:
                agent.set_obs(list_obs[agent.id])
                agent.action_sampling()
                list_act_hot.append(agent.get_action_hot())
                list_act.append(agent.get_action())

            list_rewards = []
            total_reward_given_to_each_agent = torch.zeros(env.n_agents)

            # give rewards
            for agent in agents:
                reward = agent.give_reward(list_act_hot)
                reward_sum = torch.zeros(1)
                for idx in range(env.n_agents):
                    if idx != agent.id:
                        total_reward_given_to_each_agent[idx] += reward[idx]
                        reward_sum += reward[idx]  # 计算自己总共给予了多少报酬
                reward_sum = reward_sum.detach().numpy()
                list_rewards.append(reward_sum)

                if agent.id == 0:
                    reward_one_to_two.append(reward_sum)
                else:
                    reward_two_to_one.append(reward_sum)


            # execute step
            list_obs_next, env_rewards, done = env.step(list_act, list_rewards)

            for agent in agents:
                reward_given = total_reward_given_to_each_agent[agent.id]
                trajs_new[agent.id].add(agent.get_obs(), agent.get_action(), agent.get_action_hot(), env_rewards[agent.id], reward_given)

            result_one_new += env_rewards[0]
            result_two_new += env_rewards[1]

            if done:
                results_one.append(result_one_new)
                results_two.append(result_two_new)

        log_prob_act_other = [[] for _ in range(config.env.n_agents)]
        for agent in agents:
            states_new = [trajectory.get_state() for trajectory in trajs_new]
            actions_new = [trajectory.get_action() for trajectory in trajs_new]
            logits, _ = agent.policy_net(states_new[agent.id])
            log_prob = F.log_softmax(logits, dim=-1)
            log_prob_act = torch.stack([log_prob[i][actions_new[agent.id][i]]
                                        for i in range(len(actions_new[agent.id]))],
                                       dim=0)
            log_prob_act_other[agent.id] = log_prob_act

        for agent in agents:
            agent.update_rewards_giving(trajs, trajs_new, log_prob_act_other)

    return results_one, results_two, reward_one_to_two, reward_two_to_one


def run_epoch():
    # TODO
    pass


if __name__ == "__main__":
    config = config_room_lio.get_config()

    results_one, result_two, reward1, reward2 = train(config)

    plt.figure(1)
    plt.subplot(211)
    plt.plot(results_one)
    plt.plot(result_two)
    plt.subplot(212)
    plt.plot(reward1)
    plt.plot(reward2)
    plt.show()
