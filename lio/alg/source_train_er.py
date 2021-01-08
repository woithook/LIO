"""
Trains LIO agents on Escape Room game

Implement LIO on the top of policy gradient first
"""

from __future__ import division
from __future__ import print_function

import argparse
import json
import os
import random

import numpy as np
import tensorflow as tf

from lio.alg import config_ipd_lio
from lio.alg import config_room_lio
from lio.alg import evaluate
from lio.env import room_symmetric


def train(config):

    # set seeds for the train
    seed = config.main.seed
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)

    # namespace
    dir_name = config.main.dir_name
    exp_name = config.main.exp_name
    log_path = os.path.join('..', 'results', exp_name, dir_name)
    model_name = config.main.model_name
    save_period = config.main.save_period

    # create folder for results
    os.makedirs(log_path, exist_ok=True)

    # record parameter of this run
    with open(os.path.join(log_path, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4, sort_keys=True)

    # set hyper-parameters
    n_episodes = int(config.alg.n_episodes)
    n_eval = config.alg.n_eval
    period = config.alg.period

    # parameters for epsilon-greedy method
    epsilon = config.lio.epsilon_start
    epsilon_step = (
        epsilon - config.lio.epsilon_end) / config.lio.epsilon_div

    # make env
    env = room_symmetric.Env(config.env)

    from lio_agent import LIO

    # init lio agents
    list_agents = []
    for agent_id in range(env.n_agents):
        list_agents.append(LIO(config.lio, env.l_obs, env.l_action,
                               config.nn, 'agent_%d' % agent_id,
                               config.env.r_multiplier, env.n_agents,
                               agent_id))

    # init training optimizers
    for agent_id in range(env.n_agents):
        list_agents[agent_id].receive_list_of_agents(list_agents)
        list_agents[agent_id].create_policy_gradient_op()
        list_agents[agent_id].create_update_op()

    for agent_id in range(env.n_agents):
        list_agents[agent_id].create_reward_train_op()

    # This handles the special case of two asymmetric agents,
    # one of which is the reward-giver and the other is the recipient
    if config.lio.asymmetric:
        assert config.env.n_agents == 2
        for agent_id in range(env.n_agents):
            list_agents[agent_id].set_can_give(
                agent_id != config.lio.idx_recipient
            )

    config_proto = tf.ConfigProto()
    if config.main.use_gpu:
        config_proto.device_count['GPU'] = 1
        config_proto.gpu_options.allow_growth = True
    else:
        config_proto.device_count['GPU'] = 0
    sess = tf.Session(config=config_proto)
    sess.run(tf.global_variables_initializer())

    #
    list_agent_meas = []
    list_suffix = ['reward_total', 'n_lever', 'n_door',
                   'received', 'given', 'r-lever', 'r-start', 'r-door']
    for agent_id in range(1, env.n_agents + 1):
        for suffix in list_suffix:
            list_agent_meas.append('A%d_%s' % (agent_id, suffix))

    # save models
    saver = tf.train.Saver(max_to_keep=config.main.max_to_keep)

    header = 'spisode, step_train, step,'
    header += ','.join(list_agent_meas)
    header += ',step_per_eps\n'

    with open(os.path.join(log_path, 'log.csv'), 'w') as f:
        f.write(header)

    # episode start
    step = 0
    step_train = 0
    for idx_episode in range(1, n_episodes + 1):

        # generate a trajectory
        list_buffers = run_episode(sess, env, list_agents, epsilon,
                                   prime=False)
        step += len(list_buffers[0].obs)

        for idx, agent in enumerate(list_agents):
            agent.update(sess, list_buffers[idx], epsilon)

        # generate new trajectory with new parameters of incentive function
        list_buffers_new = run_episode(sess, env, list_agents,
                                       epsilon, prime=True)
        step += len(list_buffers_new[0].obs)

        # train incentive function
        for agent in list_agents:
            if agent.can_give:
                agent.train_reward(sess, list_buffers,
                                   list_buffers_new, epsilon)

        for idx, agent in enumerate(list_agents):
            agent.update_main(sess)

        step_train += 1

        # add results to the result file
        if idx_episode % period == 0:
            (reward_total, n_move_lever, n_move_door, rewards_received,
             rewards_given, steps_per_episode, r_lever,
             r_start, r_door) = evaluate.test_room_symmetric(
                n_eval, env, sess, list_agents)
            matrix_combined = np.stack([reward_total, n_move_lever, n_move_door,
                                        rewards_received, rewards_given,
                                        r_lever, r_start, r_door])

            s = '%d,%d,%d' % (idx_episode, step_train, step)
            for idx in range(env.n_agents):
                s += ','
                s += ('{:.3e},{:.3e},{:.3e},{:.3e},{:.3e},'
                      '{:.3e},{:.3e},{:.3e}').format(
                      *matrix_combined[:, idx])
            s += ',%.2f\n' % steps_per_episode
            with open(os.path.join(log_path, 'log.csv'), 'a') as f:
                f.write(s)

        if idx_episode % save_period == 0:
            saver.save(sess, os.path.join(log_path, '%s.%d'%(
                model_name, idx_episode)))

        if epsilon > config.lio.epsilon_end:
            epsilon -= epsilon_step

    saver.save(sess, os.path.join(log_path, model_name))


def run_episode(sess, env, list_agents, epsilon, prime=False):
    list_buffers = [Buffer(env.n_agents) for _ in range(env.n_agents)]
    list_obs = env.reset()
    done = False

    while not done:
        # decide action
        list_actions = []
        for agent in list_agents:
            action = agent.run_actor(list_obs[agent.agent_id], sess,
                                     epsilon, prime)
            list_actions.append(action)

        list_rewards = []
        total_reward_given_to_each_agent = np.zeros(env.n_agents)

        # give reward
        for agent in list_agents:
            if agent.can_give:
                reward = agent.give_reward(list_obs[agent.agent_id],
                                           list_actions, sess)
            else:
                reward = np.zeros(env.n_agents)
            reward[agent.agent_id] = 0
            total_reward_given_to_each_agent += reward
            reward = np.delete(reward, agent.agent_id)
            list_rewards.append(reward)

        # execute step
        list_obs_next, env_rewards, done = env.step(list_actions, list_rewards)

        for idx, buf in enumerate(list_buffers):
            buf.add([list_obs[idx], list_actions[idx], env_rewards[idx],
                     list_obs_next[idx], done])
            buf.add_r_from_others(total_reward_given_to_each_agent[idx])
            buf.add_action_all(list_actions)
            if list_agents[idx].include_cost_in_chain_rule:
                buf.add_r_given(np.sum(list_rewards[idx]))

        list_obs = list_obs_next

    return list_buffers


class Buffer(object):
    def __init__(self, n_agents):
        self.n_agents = n_agents
        self.reset()

    def reset(self):
        self.obs = []
        self.action = []
        self.reward = []
        self.obs_next = []
        self.done = []
        self.r_from_others = []
        self.r_given = []
        self.action_all = []

    def add(self, transition):
        self.obs.append(transition[0])
        self.action.append(transition[1])
        self.reward.append(transition[2])
        self.obs_next.append(transition[3])
        self.done.append(transition[4])

    def add_r_from_others(self, r):
        self.r_from_others.append(r)

    def add_action_all(self, list_actions):
        self.action_all.append(list_actions)

    def add_r_given(self, r):
        self.r_given.append(r)


if __name__ == "__main__":
    print('Begin training...')
    config = config_room_lio.get_config()
    train(config)
    print('Finish training')
