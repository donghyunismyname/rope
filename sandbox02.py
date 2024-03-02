import torch
import numpy as np
import ipdb


x = torch.arange(2*3*4).reshape(2, 3, 4)
print(x)

x = x.view(x.shape[:-1] + torch.Size((2, x.shape[-1] // 2)))
x1, x2 = x.unbind(dim=-2)
res_torch = torch.cat((-x2, x1), dim=-1)

print(res_torch)

