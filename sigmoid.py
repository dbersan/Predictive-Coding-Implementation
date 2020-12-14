import math

# activation: sigmoid
def F(x):
    return 1 / (1 + math.exp(-x))

# derivative of activation
def dF(x):
    return F(x) * (1-F(x))

X = 0.72
print(f"{F(X)}")
print(f"{dF(X)}")