import torch
from torch import nn
from torchmeta.modules import (MetaModule, MetaLinear, MetaSequential)
from lio.utils.util import grad_graph


class MetaNet_AC(MetaModule):
    def __init__(self, l_obs, n_action, l1=64, l2=64):
        super(MetaNet_AC, self).__init__()
        self.l_obs = l_obs
        self.n_action = n_action
        self.l1 = l1
        self.l2 = l2
        self.actor_net = MetaSequential(
            MetaLinear(self.l_obs, self.l1),
            nn.ReLU(),
            MetaLinear(self.l1, self.l2),
            nn.ReLU(),
            MetaLinear(self.l2, self.n_action),
        )

        self.critic_net = MetaSequential(
            MetaLinear(self.l_obs, self.l1),
            nn.ReLU(),
            MetaLinear(self.l1, self.l2),
            nn.ReLU(),
            MetaLinear(self.l2, 1),
        )

    def forward(self, inputs, params=None):
        pi_out = self.actor_net(inputs, params=self.get_subdict(params, 'actor_net'))
        v_out = self.critic_net(inputs, params=self.get_subdict(params, 'critic_net'))
        return pi_out, v_out


class MetaNet_PG(MetaModule):
    def __init__(self, l_obs, n_action, l1=64, l2=64):
        super(MetaNet_PG, self).__init__()
        self.l_obs = l_obs
        self.n_action = n_action
        self.l1 = l1
        self.l2 = l2
        self.actor_net = MetaSequential(
            MetaLinear(self.l_obs, self.l1),
            nn.ReLU(),
            MetaLinear(self.l1, self.l2),
            nn.ReLU(),
            MetaLinear(self.l2, self.n_action),
            nn.Sigmoid(),
            MetaLinear(self.n_action, self.n_action)
        )

    def forward(self, inputs, params=None):
        pi_out = self.actor_net(inputs, params=self.get_subdict(params, 'actor_net'))
        return pi_out


def weight_init_meta(m):
    if type(m) == MetaLinear:
        nn.init.normal_(m.weight, 0.0, 0.02)
        # nn.init.xavier_normal_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



