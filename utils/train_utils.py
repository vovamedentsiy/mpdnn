import os
import torch
import numpy as np

def save_state(save_path,**kwargs):
    torch.save(kwargs,save_path)