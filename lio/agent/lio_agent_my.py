import copy
import numpy as np
import torch
from torch import optim
import torch.nn.functional as F
from torch.distributions import Categorical
import torchviz

import lio.model.actor_net as actor_net
from lio.model.actor_net import Net
from lio.model.actor_net import Reward_net
from lio.utils import util


gamma = 0.99
lr_a = 0.01


class Actor(object):
    def __init__(self, agent_id, l_obs, n_agent, l1=64, l2=32, lr=1e-4):
        self.id = agent_id
        self.n_action = 3
        self.l_act = 3
        self.l_obs = l_obs
        self.n_agent = n_agent

        self.l1 = l1
        self.l2 = l2
        self.lr = lr

        self.obs = []
        self.action = []
        self.action_hot = []
        self.reward_given = []

        self.policy_net = Net(self.l_obs, self.n_action, l1, l2)
        self.reward_net = Reward_net(self.l_obs, self.l_act, n_agent, l1, 16)
        self.policy_net.apply(actor_net.weight_init)
        self.reward_net.apply(actor_net.weight_init)

        self.optimizer_p = optim.Adam(self.policy_net.parameters(), lr=self.lr)
        self.optimizer_r = optim.Adam(self.reward_net.parameters(), lr=1e-3)

    def reset_state(self):
        self.obs = []
        self.action = []
        self.action_hot = []
        self.reward_given = []

    def set_obs(self, obs):
        self.obs = obs

    def get_obs(self):
        return self.obs

    def action_sampling(self):
        with torch.no_grad():
            obs = torch.Tensor(self.obs)
            logits, _ = self.policy_net(obs)
            probs = F.softmax(logits, dim=0)
            m = Categorical(probs)
            action_prim = m.sample()
            action_hot = np.zeros(self.l_act)
            action_hot[action_prim] = 1
            self.action = action_prim
            self.action_hot = action_hot

    def get_action(self):
        return self.action

    def get_action_hot(self):
        return self.action_hot

    def give_reward(self, list_action):
        action_other = copy.copy(list_action)
        del action_other[self.id]
        action_other = torch.Tensor(action_other).view(-1)
        obs = torch.Tensor(self.obs).view(-1)
        input = torch.cat([obs, action_other])
        reward = self.reward_net(input)
        return reward

    def set_reward_given(self, reward_given):
        self.reward_given = reward_given

    def get_reward_given(self):
        return self.reward_given

    def update_policy(self, trajectory):
        # self.optimizer_p.zero_grad()
        self.policy_net.zero_grad()

        states = trajectory.get_state()
        actions = trajectory.get_action()
        rewards_env = trajectory.get_reward_env()
        rewards_given = trajectory.get_reward_given()
        loss_policy = []
        loss_critic = []
        loss_entropy = []

        R = 0
        returns_env = []
        returns_given = []

        # Compute policy loss
        # Get the V value of timestep from critic
        logits, V_s = self.policy_net(states)
        prob = F.softmax(logits, dim=-1)
        log_prob = F.log_softmax(logits, dim=-1)
        V_s = V_s.view(-1)

        for r in rewards_env[::-1]:
            R = r + gamma * R
            returns_env.insert(0, R)
        for r in rewards_given[::-1]:
            R = r + gamma * R
            returns_given.insert(0, R)

        returns_env = torch.Tensor(returns_env).detach()
        returns_given = torch.stack(returns_given, dim=0)
        returns = returns_env + returns_given
        # returns = returns_env
        Q_s_a = returns

        A_s_a = Q_s_a - V_s

        # compute policy loss
        loss_entropy_p = - log_prob * prob
        loss_entropy_p = loss_entropy_p.mean()
        loss_entropy.append(loss_entropy_p)

        log_prob_act = torch.stack([log_prob[i][actions[i]] for i in range(len(actions))], dim=0)
        loss_policy_p = - torch.dot(A_s_a, log_prob_act).view(1) / len(prob)
        loss_policy.append(loss_policy_p)

        # Compute critic loss
        # loss_critic_p = (returns - V_s).pow(2).mean()
        loss_critic_p = A_s_a.pow(2).mean()
        loss_critic.append(loss_critic_p)

        loss_policy = torch.stack(loss_policy).mean()
        loss_critic = torch.stack(loss_critic).mean()
        loss_entropy = torch.stack(loss_entropy).mean()
        loss = loss_policy + 0.5 * loss_critic + 0.001 * loss_entropy
        # loss = loss_policy + loss_critic

        loss.backward(retain_graph=True)
        # for p in self.policy_net.parameters():
        #     new_val = p - lr_a * p.grad
        #     p.copy_(new_val)
        self.optimizer_p.step()

    def update_rewards_giving(self, trajectorys, trajectorys_new, log_prob_act_other):
        # util.grad_graph(self.policy_net.l1.weight)
        self.optimizer_r.zero_grad()

        loss_policy = 0
        loss_cost = 0

        states = [trajectory.get_state() for trajectory in trajectorys]
        actions_hot = [trajectory.get_action_hot() for trajectory in trajectorys]
        n_step = len(states[0])

        rewards_env_new = [trajectory.get_reward_env() for trajectory in trajectorys_new]

        # compute loss of cost
        n_agent = self.n_agent - 1
        actions_other = copy.copy(actions_hot)
        del actions_other[self.id]
        actions_other = [[actions_other[j][i] for j in range(n_agent)] for i in range(n_step)]
        actions_other = [torch.tensor(actions_other[i]).view(-1) for i in range(n_step)]
        obs = states[self.id]
        feeds_r = [torch.cat([obs[i], actions_other[i]]) for i in range(n_step)]
        feeds_r = torch.stack(feeds_r).float()
        rewards_giving = self.reward_net(feeds_r)
        rewards_giving_total = torch.zeros(n_step)
        for i in range(n_step):
            rewards_giving_total[i] = (gamma ** i) * (rewards_giving[i][torch.arange(rewards_giving.size(1))!=self.id].sum())
        loss_cost = rewards_giving_total.sum()

        # compute policy loss
        returns_env = []
        R = 0

        reward_env = rewards_env_new[self.id]
        for r in reward_env[::-1]:
            R = r + gamma * R
            returns_env.insert(0, R)
        returns_env = torch.tensor(returns_env).float()

        for idx in range(self.n_agent):
            if idx != self.id:
                loss_term = log_prob_act_other[idx]
                loss_policy += - torch.dot(returns_env.detach(), loss_term).view(-1)

        loss = loss_policy + loss_cost

        util.grad_graph(loss_policy)
        loss.backward()
        # for name, parms in self.policy_net.named_parameters():
        #     print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
        #           ' \n-->grad_value:', parms.grad)
        # print('----------------------------------------------------\n\n')
        self.optimizer_r.step()







