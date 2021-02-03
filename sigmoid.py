import torch
import torch.nn
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
m[0,0] = -1
print(m)
print(m*m)

v = torch.tensor([0.1,0.01,1,2,3])
zeros = torch.zeros(5, dtype=dtype)
c = torch.lt(zeros, v).prod()
print( c )
if c:
    print('aa')









# print(m.transpose(1,0))