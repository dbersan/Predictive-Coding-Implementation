import torch
import math
import numpy as np

dtype = torch.float

x = torch.linspace(-math.pi, math.pi, 2000, dtype=dtype)
y = torch.sin(x)

a = torch.rand(4,2, dtype=dtype)
b = np.random.rand(2,3)
b = torch.from_numpy(b.astype(np.float32))
m = torch.matmul(a,b)



print(a)
print(b)
print(m)


