import copy
import numpy as np
from collections import OrderedDict
import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
from torchmeta.utils.gradient_based import gradient_update_parameters

import lio.model.meta_actor_net as meta_actor_net
import lio.model.actor_net as actor_net
from lio.model.meta_actor_net import MetaNet_PG
from lio.model.actor_net import Reward_net

import lio.utils.util as util
from lio.utils.util import grad_graph
from lio.utils.util import check_reward_net_grad
from lio.utils.util import Adam_Optim


gamma = 0.99
eps = torch.finfo(torch.float32).eps


class Actor(object):
    def __init__(self, agent_id, l_obs, n_agent, n_action=3, l_act=3, l1=64, l2=32, lr=0.01):
        self.id = agent_id
        self.n_action = n_action
        self.l_act = l_act
        self.l_obs = l_obs
        self.n_agent = n_agent
        self.lr = lr

        self.l1 = l1
        self.l2 = l2

        self._obs = []
        self._action = []
        self._action_hot = []
        self._reward_from = []

        self.policy_net = MetaNet_PG(self.l_obs, self.n_action, l1, l2)
        self.reward_net = Reward_net(self.l_obs, self.l_act, n_agent, l1, 16)
        self.policy_net.apply(meta_actor_net.weight_init_meta)
        self.reward_net.apply(actor_net.weight_init)
        self.new_params = None

        self.optimizer_p = Adam_Optim(self.policy_net, lr=self.lr)
        self.optimizer_r = optim.Adam(self.reward_net.parameters(), lr=1e-2)

    def reset_state(self):
        self._obs = []
        self._action = []
        self._action_hot = []
        self._reward_from = []
        self.new_params = []

    def set_obs(self, obs):
        self._obs = obs

    def get_obs(self):
        return self._obs

    def action_sampling(self, epsilon):
        with torch.no_grad():
            obs = torch.Tensor(self._obs)
            logits = self.policy_net(obs, self.new_params)
            probs = F.softmax(logits, dim=0)
            probs_hat = (1 - epsilon) * probs + epsilon / self.n_action
            m = Categorical(probs_hat)
            try:
                action_prim = m.sample()
            except RuntimeError:
                print("logits = ", logits, "\nprobs = ", probs, "\nprobs_hat = ", probs_hat)
            action_hot = np.zeros(self.l_act)
            action_hot[action_prim] = 1
            self._action = action_prim
            self._action_hot = action_hot

    def get_action(self):
        return self._action

    def get_action_hot(self):
        return self._action_hot

    def give_reward(self, list_action):
        action_other = copy.copy(list_action)
        del action_other[self.id]
        action_other = torch.Tensor(action_other).view(-1)
        obs = torch.Tensor(self._obs).view(-1)
        input = torch.cat([obs, action_other]).detach()
        reward = self.reward_net(input)
        reward = reward * 1
        return reward

    def set_reward_given(self, reward_given):
        self._reward_from = reward_given

    def get_reward_from(self):
        return self._reward_from

    def update_policy(self, trajectorys, lr=0.1):
        self.policy_net.zero_grad()

        states = trajectorys[self.id].get_state()
        actions = trajectorys[self.id].get_action()

        returns_env = trajectorys[self.id].get_returns_env()
        returns_from = trajectorys[self.id].get_returns_from()

        returns_env = torch.Tensor(returns_env)
        returns_from = torch.Tensor(returns_from)
        returns = torch.add(returns_env, returns_from)
        # returns = returns_env

        # Compute policy loss
        logits = self.policy_net(states)
        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)

        # compute policy loss
        loss_entropy = - log_prob * prob
        loss_entropy = loss_entropy.mean()

        log_prob_act = torch.stack([log_prob[i][actions[i]] for i in range(len(actions))], dim=0)
        loss_policy = - torch.dot(returns, log_prob_act).view(1) / len(prob)

        loss = loss_policy + 0.01 * loss_entropy
        self.new_params = util.gd(self.policy_net, loss, lr=lr)

    def update_policy_op(self, trajectorys):
        self.policy_net.zero_grad()

        states = trajectorys[self.id].get_state()
        actions = trajectorys[self.id].get_action()
        rewards_env = trajectorys[self.id].get_reward_env()
        rewards_from = trajectorys[self.id].get_reward_from()
        loss_policy = []
        loss_entropy = []

        returns_env = []
        returns_from = []

        # Compute policy loss
        logits = self.policy_net(states)
        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)

        R = 0
        for r in rewards_env[::-1]:
            R = r + gamma * R
            returns_env.insert(0, R)
        R = 0
        for r in rewards_from[::-1]:
            R = r + gamma * R
            returns_from.insert(0, R)

        returns_env = torch.Tensor(returns_env).detach()
        # returns_from = torch.cat(returns_from, dim=0)
        # returns = returns_env + returns_from
        returns = returns_env
        if len(returns) != 1:
            returns = (returns - returns.mean()) / (returns.std() + eps)
        else:
            returns -= returns.mean()

        # compute policy loss
        loss_entropy_p = - log_prob * prob
        loss_entropy = loss_entropy_p.mean()

        log_prob_act = torch.stack([log_prob[i][actions[i]] for i in range(len(actions))], dim=0)
        loss_policy = - torch.dot(returns, log_prob_act).view(1) / len(prob)

        loss = loss_policy + 0.001 * loss_entropy

        self.new_params = self.optimizer_p.update(self.policy_net, loss, retain_graph=True)

    def update_to_new_params(self):
        if self.new_params is None:
            raise ValueError("The policy has not been updated. "
                             "Please check that you had use 'update_policy()' before.")
        with torch.no_grad():
            for p, new_p in zip(self.policy_net.parameters(), self.new_params.items()):
                p.copy_(new_p[1])
        self.new_params = None

    def update_rewards_giving(self, trajectorys, trajectorys_new, log_prob_act_other):
        self.optimizer_r.zero_grad()

        loss_policy = torch.Tensor([0])
        loss_cost = None

        states = [trajectory.get_state() for trajectory in trajectorys]
        actions_hot = [trajectory.get_action_hot() for trajectory in trajectorys]
        n_step = len(states[0])

        # compute loss of cost
        n_agent = self.n_agent - 1
        actions_other = copy.copy(actions_hot)
        del actions_other[self.id]
        actions_other = [[actions_other[j][i] for j in range(n_agent)] for i in range(n_step)]
        actions_other = [torch.Tensor(actions_other[i]).view(-1) for i in range(n_step)]
        obs = states[self.id]
        feeds_r = [torch.cat([obs[i], actions_other[i]]) for i in range(n_step)]
        feeds_r = torch.stack(feeds_r).float()
        rewards_giving = self.reward_net(feeds_r)
        rewards_giving_total = torch.zeros(n_step)
        for i in range(n_step):
            rewards_giving_total[i] = (gamma ** i) * (rewards_giving[i][torch.arange(rewards_giving.size(1)) != self.id].sum())
        loss_cost = rewards_giving_total.sum()

        # compute policy loss
        rewards_env_new = [trajectory.get_reward_env() for trajectory in trajectorys_new]
        returns_env = []
        R = 0
        reward_env = rewards_env_new[self.id]
        for r in reward_env[::-1]:
            R = r + gamma * R
            returns_env.insert(0, R)
        returns_env = torch.Tensor(returns_env).float()
        if len(returns_env) != 1:
            returns_env = (returns_env - returns_env.mean()) / (returns_env.std() + eps)
        else:
            returns_env = returns_env - returns_env.mean() + eps

        for idx in range(self.n_agent):
            if idx != self.id:
                loss_term = log_prob_act_other[idx]
                loss_policy -= torch.dot(returns_env.detach(), loss_term).view(-1)

        # loss = loss_policy + 0.0001 * loss_cost
        loss = loss_policy

        loss.backward()

        # print('for agent ', self.id, ', the gradient of its parameters is as following:')
        # for name, parms in self.reward_net.named_parameters():
        #     if name == "l3.bias":
        #         print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
        #               ' \n-->grad_value:', parms.grad)
        # print('----------------------------------------------------\n\n')
        self.optimizer_r.step()
