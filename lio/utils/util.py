import numpy as np
import torchviz
import torch
import math

from collections import OrderedDict
from torchmeta.modules import MetaModule

eps = torch.finfo(torch.float32).eps


class Adam_Optim(object):
    def __init__(self, model, lr=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
        params = OrderedDict(model.named_parameters())
        self.m_t = dict()
        self.v_t = dict()
        for name, param in params.items():
            self.m_t[name] = torch.zeros(param.shape)
            self.v_t[name] = torch.zeros(param.shape)
        self.t = 0
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.lr = lr

    def update(self, model, loss, retain_graph=False):
        self.t += 1

        if not isinstance(model, MetaModule):
            raise ValueError('The model must be an instance of `torchmeta.modules.'
                             'MetaModule`, got `{0}`'.format(type(model)))

        # loss.backward(retain_graph=retain_graph)
        params = OrderedDict(model.named_parameters())
        grads = torch.autograd.grad(loss, params.values(), create_graph=True)
        updated_params = OrderedDict()

        # with torch.no_grad():
        for (name, param), grad in zip(params.items(), grads):
            # self.m_t[name] = self.m_t[name].detach()
            # self.v_t[name] = self.v_t[name].detach()

            self.m_t[name] = self.beta1 * self.m_t[name] + (1 - self.beta1) * grad
            self.v_t[name] = self.beta2 * self.v_t[name] + (1 - self.beta2) * (grad ** 2)
            m_cap = self.m_t[name] / (1 - self.beta1 ** self.t)
            v_cap = self.v_t[name] / (1 - self.beta2 ** self.t)
            v_bias = torch.sqrt(v_cap + eps) + self.epsilon
            updated_params[name] = param - self.lr * (m_cap / v_bias)

        return updated_params

    def reset(self):
        for m, v in zip(self.m_t, self.v_t):
            m = 0
            v = 0
        self.t = 0


def gd(model, loss, lr=0.1):
    if not isinstance(model, MetaModule):
        raise ValueError('The model must be an instance of `torchmeta.modules.'
                         'MetaModule`, got `{0}`'.format(type(model)))

    params = OrderedDict(model.named_parameters())
    grads = torch.autograd.grad(loss, params.values(), create_graph=True)
    updated_params = OrderedDict()

    for (name, param), grad in zip(params.items(), grads):
        updated_params[name] = param - lr * grad
        grad_graph(grad, 'grad')

    return updated_params


def grad_graph(var, file_name=None):
    dot = torchviz.make_dot(var)
    dot.format = 'svg'
    dot.render(file_name)


def check_reward_net_grad(val, agents, id=None):
    from lio.agent.lio_agent_pg import Actor

    if agents is isinstance(agents, Actor):
        val.backward()
        print('for agent ', agents.id, ', the gradient of its parameters is as following:')
        for name, parms in agents.reward_net.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                  ' \n-->grad_value:', parms.grad)
        print('----------------------------------------------------\n\n')
        agents.reward_net.zero_grad()
    else:
        val.backward()
        print('for agent ', agents[id].id, ', the gradient of its parameters is as following:')
        for name, parms in agents[id].reward_net.named_parameters():
            print('-->name:', name, '-->grad_requirs:', parms.requires_grad,
                  ' \n-->grad_value:', parms.grad)
        print('----------------------------------------------------\n\n')
        agents[id].reward_net.zero_grad()


def process_actions(actions, l_action):

    n_steps = len(actions)
    actions_1hot = np.zeros([n_steps, l_action], dtype=int)
    actions_1hot[np.arange(n_steps), actions] = 1

    return actions_1hot


def get_action_others_1hot(action_all, agent_id, l_action):
    action_all = list(action_all)
    del action_all[agent_id]
    num_others = len(action_all)
    actions_1hot = np.zeros([num_others, l_action], dtype=int)
    actions_1hot[np.arange(num_others), action_all] = 1

    return actions_1hot.flatten()


def get_action_others_1hot_batch(list_action_all, agent_id, l_action):
    n_steps = len(list_action_all)
    n_agents = len(list_action_all[0])
    matrix = np.stack(list_action_all)  # [n_steps, n_agents]
    self_removed = np.delete(matrix, agent_id, axis=1)
    actions_1hot = np.zeros([n_steps, n_agents-1, l_action], dtype=np.float32)
    grid = np.indices((n_steps, n_agents-1))
    actions_1hot[grid[0], grid[1], self_removed] = 1
    actions_1hot = np.reshape(actions_1hot, [n_steps, l_action*(n_agents-1)])

    return actions_1hot


def process_rewards(rewards, gamma):
    n_steps = len(rewards)
    gamma_prod = np.cumprod(np.ones(n_steps) * gamma)
    returns = np.cumsum((rewards * gamma_prod)[::-1])[::-1]
    returns = returns / gamma_prod

    return returns
