import torch
from torch.nn import init
import numpy as np


BN_COUNTER=0
class _BatchNorm(torch.nn.modules.batchnorm._BatchNorm):
    def __init__(self, name=None,*args, **kwargs):
        super(_BatchNorm, self).__init__(*args, **kwargs)

        if name is None:
            global  BN_COUNTER
            BN_COUNTER +=1
            name = "BatchNorm{}".format(BN_COUNTER)
        self.name=name

    def get_tb_summaries(self):
        tb_summaries = {}
        tb_summaries[self.name + "/gamma"] = self.weight
        tb_summaries[self.name + "/beta"] = self.bias
        tb_summaries[self.name + "/running_mean"] = self.running_mean
        tb_summaries[self.name + "/running_var"] = self.running_var
        return tb_summaries

    def _get_minibatch_statistics(self, x, update_running_stats=True):
        print('get minibatch stats ')


        reduction_indices = (0, 2, 3) if len(x.shape) == 4 else (0,)
        mb_mean = torch.mean(x, dim=reduction_indices, keepdim=True)
        mb_var = (torch.sum((x - mb_mean) ** 2, dim=reduction_indices)) / (
                np.prod([x.shape[i] for i in reduction_indices]) - 1)

        mb_mean = mb_mean.squeeze()
        if update_running_stats:
            with torch.no_grad():
                self.running_mean.data = (1 - self.momentum) * self.running_mean.data + self.momentum * mb_mean
                self.running_var.data = (1 - self.momentum) * self.running_var.data + self.momentum * mb_var

        return mb_mean, mb_var

    def reset_parameters(self):

        self.reset_running_stats()
        if self.affine:
            init.ones_(self.weight)
            init.zeros_(self.bias)

    def forward(self, x):

        self.input = x
        out = super(_BatchNorm, self).forward(x)
        self.output=out

        return out


class BatchNorm2d(_BatchNorm):
    def __init__(self,*args,**kwargs):
        super(BatchNorm2d,self).__init__(*args,**kwargs)
    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

class BatchNorm1d(_BatchNorm):
    def __init__(self,*args,**kwargs):
        super(BatchNorm1d,self).__init__(*args,**kwargs)
    def _check_input_dim(self, input):
        if input.dim() != 2 and input.dim() != 3:
            raise ValueError('expected 2D or 3D input (got {}D input)'
                             .format(input.dim()))