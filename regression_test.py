import matplotlib.pyplot as plt
import numpy as np
from snn.FreeEnergyNetwork import FreeEnergyNetwork

# Network parameters
neurons             = 2
layers              = 3
neurons_per_layer   = [2, 3, 2]

# Training parameters
inference_steps     = 40
weight_update_steps = 1
inner_loop_count    = 5