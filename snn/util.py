import torch.nn
import torch

def dRelu(x):
    return torch.gt(x, torch.zeros(x.shape)).type(x.type())

def dSigmoid(x):
    sig_x = torch.nn.Sigmoid()(x)
    return sig_x*(1 - sig_x)
