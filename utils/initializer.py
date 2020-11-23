from torch.nn.init import _calculate_correct_fan
import math
import torch

import numpy as np
from scipy.stats import truncnorm
import torch


def keras_replica(tensor):
    fan = _calculate_correct_fan(tensor, "fan_in")
    scale = 1.0
    scale /= max(1., fan)
    stddev = math.sqrt(scale) / .87962566103423978

    init_val = (truncnorm.rvs(size=tensor.shape, a=-2, b=2) * stddev).astype(np.float32)
    with torch.no_grad():
        tensor.data = torch.as_tensor(init_val)

if __name__ == '__main__':
    x = torch.zeros((10,10))
    keras_replica(x)