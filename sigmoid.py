import torch
import math
import numpy as np
import copy

dtype = torch.float

x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)
y = torch.sin(x)

a = torch.rand(4,2)
a = torch.rand(4,2, dtype=np.float)
b = np.random.rand(2,3).astype(np.float)
b = torch.from_numpy(b)
m = torch.matmul(a,b)



print(a)
print(b)
print(m)


a = np.array([[1,2,3],[4,5,6]])
b = a.reshape(-1,1).astype(np.float)
a[0,0] = -7777
print(b)