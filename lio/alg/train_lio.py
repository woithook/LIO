"""
Reproduction of LIO in the Escape Room
"""

import numpy as np
import random
import torch
from matplotlib import pyplot as plt

from lio.agent.lio_agent_my import Actor
from lio.model.actor_net import Trajectory

from lio.alg import config_room_lio
from lio.env import room_symmetric


def train(config):

    # set seeds for the training
    # seed = config.main.seed
    seed = 12345
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
    for epoch in range(6000):
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
            total_reward_given_to_each_agent = np.zeros(env.n_agents)

            # give rewards
            for agent in agents:
                reward = agent.give_reward(list_act_hot)
                reward[agent.id] = 0
                total_reward_given_to_each_agent += reward
                reward = np.delete(reward, agent.id)
                list_rewards.append(reward)

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
            total_reward_given_to_each_agent = np.zeros(env.n_agents)

            # give rewards
            for agent in agents:
                reward = agent.give_reward(list_act_hot)
                reward[agent.id] = 0
                total_reward_given_to_each_agent += reward
                reward = np.delete(reward, agent.id)
                list_rewards.append(reward)
                if agent.id == 0:
                    reward_one_to_two.append(reward)
                else:
                    reward_two_to_one.append(reward)


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

        for agent in agents:
            agent.update_rewards_giving(trajs, trajs_new)

    return results_one, results_two, reward_one_to_two, reward_two_to_one


def run_epoch():
    # TODO
    pass


if __name__ == "__main__":
    config = config_room_lio.get_config()

    results_one, result_two, reward1, reward2 = train(config)
    # plt.plot(results_one)
    # plt.plot(result_two)
    plt.plot(reward1)
    plt.plot(reward2)
    plt.show()
