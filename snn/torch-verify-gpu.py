# Verifies if torch has access to GPU
# Source: https://stackoverflow.com/a/48152675/2076973

import torch

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

print(device)
