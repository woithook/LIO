from metann import DependentModule
import torch
from torch import nn
net = torch.nn.Sequential(
    nn.Linear(10, 100),
    nn.Linear(100, 5))
net = DependentModule(net)
print(net)

for p in net.parameters():
    new_val = torch.Tensor(1)
    p.copy_(new_val)
    p.grad = 0

for name, parms in net.named_parameters():
    print('-->name:', name, '\n-->grad_requirs:', parms.requires_grad,
          ' \n-->value:', parms.data)
    print('----------------------------------------------------\n\n')
