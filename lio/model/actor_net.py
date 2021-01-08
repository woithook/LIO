import torch
from torch import nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, l_obs, n_action, l1=64, l2=64):
        super(Net, self).__init__()
        self.l1 = nn.Linear(l_obs, l1)
        self.l2 = nn.Linear(l1, l2)
        self.pi = nn.Linear(l2, n_action)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()

        self.c_l1 = nn.Linear(l_obs, l1)
        self.c_l2 = nn.Linear(l1, l2)
        self.c_relu1 = nn.ReLU()
        self.c_relu2 = nn.ReLU()
        self.v = nn.Linear(l2, 1)

    def forward(self, input):
        z1 = self.relu1(self.l1(input))
        z2 = self.relu2(self.l2(z1))
        # pi_out = F.softmax(self.pi(z2), dim=0)
        pi_out = self.pi(z2)

        c_z1 = self.c_relu1(self.c_l1(input))
        c_z2 = self.c_relu2(self.c_l2(c_z1))
        v_out = self.v(c_z2)

        return pi_out, v_out


class Reward_net(nn.Module):
    # 输入t时刻自机的观察值以及其他Agent的观察值，输入报酬分配率
    # 输入的整合需要在网络外完成
    def __init__(self, l_obs, l_act, n_agent, l1=64, l2=64):
        super(Reward_net, self).__init__()
        input_size = l_obs + l_act * (n_agent - 1)
        self.l1 = nn.Linear(input_size, l1)
        self.l2 = nn.Linear(l1, l2)
        self.l3 = nn.Linear(l2, n_agent)  # 输出层
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, input):
        z1 = self.relu1(self.l1(input))
        z2 = self.relu2(self.l2(z1))
        output = self.sig(self.l3(z2))
        return output


def weight_init(m):
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, 0.0, 0.02)
        nn.init.constant_(m.bias, 0)


class Trajectory(object):
    def __init__(self):
        self.current_state = []
        self.current_action = []
        self.current_action_hot = []
        self.reward_env = []
        self.reward_given = []

    def add(self, current_state, current_action, current_action_hot, reward, reward_given):
        self.current_state.append(current_state)
        self.current_action.append(current_action)
        self.current_action_hot.append(current_action_hot)
        self.reward_env.append(reward)
        self.reward_given.append(reward_given)

    def get_state(self):
        self.current_state = torch.Tensor(self.current_state)
        return self.current_state

    def get_reward_env(self):
        return self.reward_env

    def get_reward_given(self):
        return self.reward_given

    def get_action(self):
        return self.current_action

    def get_action_hot(self):
        return self.current_action_hot



