import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# Randomly select an index to be "hot" (1) in the one-hot encoded vector
# torch.randint(low, high, size) generates a random integer within [low, high)
real_res = torch.Tensor(128 * [1] + 128 * [0])
print(real_res)