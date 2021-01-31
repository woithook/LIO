"""
Reproduction of LIO in the Escape Room

Fix the shared variables used in both two trajectory generations in one epoch.
1. 两个agent各用不同的槽（可以以属性的形式装在agent类里？
2. 两道轨迹各用不同的槽
"""

import numpy as np
import random
import torch
from torch.nn import functional as F
from matplotlib import pyplot as plt

from lio.agent.lio_agent_pg import Actor
from lio.model.actor_net import Trajectory

from lio.alg import config_room_lio
from lio.env import room_symmetric
from lio.utils.util import grad_graph
from lio.utils.util import check_reward_net_grad


def train(config):

    # set random seed
    seed = 1234
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)

    num_epoch = 5000
    lr_a = 0.01

    epsilon = 0.5
    epsilon_end = 0.1
    epsilon_div = 1e3
    epsilon_step = (
            (epsilon - epsilon_end) / epsilon_div)

    results_one = []
    results_two = []
    reward_one_to_two = []
    reward_two_to_one = []

    # init game env
    env = room_symmetric.Env(config.env)

    agents = []
    for i in range(env.n_agents):
        agents.append(Actor(i, 7, env.n_agents, lr=0.01))

    # epoch start
    for epoch in range(num_epoch):
        if (epoch + 1) % 500 == 0:
            print("Epoch: ", epoch + 1, "/", num_epoch)

        """The first trajectory generation"""
        trajs = [Trajectory() for _ in range(env.n_agents)]
        list_obs = env.reset()
        list_obs_next = None
        done = None

        while not done:
            if list_obs_next is not None:
                list_obs = list_obs_next

            # decide actions from observations
            list_act, list_act_hot = action_sampling(agents, list_obs, epsilon)

            # give incentivisation
            inctv_to, inctv_from = give_incentivisation(agents, list_act_hot, config)
            # execute step
            list_obs_next, env_rewards, done = env.step(list_act, inctv_to)

            # save trajectory
            for agent in agents:
                trajs[agent.id].add(agent.get_obs(), agent.get_action(), agent.get_action_hot(), env_rewards[agent.id],
                                    inctv_from[agent.id])

        for agent in agents:
            agent.update_policy(trajs, lr=lr_a)

        """The second trajectory generation"""
        # Generate a new trajectory
        trajs_new = [Trajectory() for _ in range(env.n_agents)]
        list_obs = env.reset()
        list_obs_next = None
        done = False
        result_one_new = 0
        result_two_new = 0

        while not done:
            if list_obs_next is not None:
                list_obs = list_obs_next
            # decide actions from observations
            list_act, list_act_hot = action_sampling(agents, list_obs, epsilon)

            # give incentivisation
            inctv_to, new_inctv_from_others = give_incentivisation(agents, list_act_hot, config)

            reward_one_to_two.append(inctv_to[0])
            reward_two_to_one.append(inctv_to[1])

            # execute step
            list_obs_next, env_rewards, done = env.step(list_act, inctv_to)

            for agent in agents:
                trajs_new[agent.id].add(agent.get_obs(), agent.get_action(), agent.get_action_hot(),
                                        env_rewards[agent.id], new_inctv_from_others[agent.id])

            result_one_new += env_rewards[0]
            result_two_new += env_rewards[1]

            if done:
                results_one.append(result_one_new)
                results_two.append(result_two_new)

        # compute new log prob act
        log_prob_act_other = compute_log_prob_act_other(agents, trajs_new, config)

        for agent in agents:
            agent.update_rewards_giving(trajs, trajs_new, log_prob_act_other)

        for agent in agents:
            agent.update_to_new_params()

        if epsilon > epsilon_end:
            epsilon -= epsilon_step

    return results_one, results_two, reward_one_to_two, reward_two_to_one


def action_sampling(agents, list_obs, epsilon=0.0):
    list_act = []
    list_act_hot = []
    for agent in agents:
        agent.set_obs(list_obs[agent.id])
        agent.action_sampling(epsilon)
        list_act.append(agent.get_action())
        list_act_hot.append(agent.get_action_hot())
    return list_act, list_act_hot


def give_incentivisation(agents, list_act_hot, config):
    list_inctv = []
    inctv_from_others = [torch.Tensor([0]) for _ in range(config.env.n_agents)]
    inctv_to_others = [None for _ in range(config.env.n_agents)]
    for agent in agents:
        inctv_to_others[agent.id] = agent.give_reward(list_act_hot)
        for idx in range(config.env.n_agents):
            if idx != agent.id:
                inctv_from_others[idx] += inctv_to_others[agent.id][idx]
        inctv_sum = (inctv_to_others[agent.id].sum()
                     - inctv_to_others[agent.id][agent.id]).detach().numpy()  # 计算自己总共给予了多少报酬
        list_inctv.append(inctv_sum)
    return list_inctv, inctv_from_others


def compute_log_prob_act_other(agents, trajectorys, config):
    log_prob_act_other = [None for _ in range(config.env.n_agents)]
    states_new = [trajectory.get_state() for trajectory in trajectorys]
    actions_new = [trajectory.get_action() for trajectory in trajectorys]
    for agent in agents:
        logits = agent.policy_net(states_new[agent.id], agent.new_params)
        check_reward_net_grad(logits.sum(), agents, 1)

        log_prob = F.log_softmax(logits, dim=-1)
        log_prob_act = torch.stack([log_prob[i][actions_new[agent.id][i]]
                                    for i in range(len(actions_new[agent.id]))], dim=0)
        log_prob_act_other[agent.id] = log_prob_act
    return log_prob_act_other


if __name__ == "__main__":
    for _ in range(1):
        config = config_room_lio.get_config()
        with torch.autograd.set_detect_anomaly(True):
            results_one, results_two, reward1, reward2 = train(config)

        col_results = [i + j for i, j in zip(results_one, results_two)]

        plt.figure(1)
        plt.subplot(211)
        plt.plot(results_one)
        plt.plot(results_two)
        # plt.plot(col_results)
        plt.subplot(212)
        plt.ylim([0, 2])
        plt.plot(reward1)
        plt.plot(reward2)
        plt.show()

