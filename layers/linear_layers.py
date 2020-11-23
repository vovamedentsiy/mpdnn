import torch
from torch.nn import init

from torch.nn import Linear, Conv2d

from layers.quantizer import Quantizer
from utils.initializer import keras_replica

import numpy as np



class _MPBaseLayer(torch.nn.Module):
    def __init__(self, quantize_weights, *args, **kwargs):
        super(_MPBaseLayer, self).__init__(*args, **kwargs)

        self.quantize_weights = quantize_weights
        self.quantize_flag = quantize_weights
        if self.quantize_weights:
            self.weight_quantizer = Quantizer()

        self.tb_summaries_train = {}
        self.tb_summaries_eval = {}




    def get_named_parameters(self):
        own_params = {self.name + "/weight": self.weight}
        if self.bias is not None:
            own_params[self.name + "/bias"] = self.bias
        for c in self.children():
            own_params.update({self.name + "/" + k: v for k, v in c.get_named_parameters().items()})
        return own_params

    def get_tb_summaries(self):
        tb_summaries = {}
        if self.training:
            tb_summaries[self.name + "/weight"] = self.weight
            if self.bias is not None:
                tb_summaries[self.name + "/bias"] = self.bias
            for k, v in self.tb_summaries_train.items():
                tb_summaries[self.name + "/" + k] = v



        else:
            for k, v in self.tb_summaries_eval.items():
                tb_summaries[self.name + "/" + k] = v

        for c in self.children():
            if hasattr(c,"get_tb_summaries"):
                tb_summaries.update({self.name + "/" + k: v for k, v in c.get_tb_summaries().items()})

        return tb_summaries

    def _get_parameters(self, x=None):
        w = self.weight
        b = self.bias

        if self.quantize_weights:
            w = self.weight_quantizer(w)

        return w, b

    @torch.no_grad()
    def plot_weights(self):

        if self.quantize_weights:
            w = self.weight_quantizer(self.weight)

        (unique, counts) = np.unique(w.detach().cpu().numpy(), return_counts=True)

        dict_frequency = {k:0 for k in range(int(2 ** self.weight_quantizer.get_bits()) - 1)}
        total_weights = counts.sum()


        with torch.no_grad():
            params_dict = self.weight_quantizer.get_params()
            d, qm = params_dict['d_hard'].item(), params_dict['qm_clamped'].item()


        for u, c in zip(unique, counts):

            k = np.round((u + qm) / d)
            dict_frequency[k] = c / total_weights

        return dict_frequency




    def train_forward(self, x):


        w, b = self._get_parameters(x)

        output = self.linear_op(x, w, b)

        return output

    def test_forward(self, x):

        w, b = self._get_parameters(x)
        output = self._linear_op(x, w, b)

        return output

    def linear_op(self, x, w, b=None):

        return self._linear_op(x, w, b)



    def forward(self, x):

        if self.training:
            out = self.train_forward(x)
        else:
            out = self.test_forward(x)
        self.output = out
        return out

    def _linear_op(self, x, w, b=None):
        raise NotImplementedError()





FC_LAYER_COUNTER = 0


class MPFC(_MPBaseLayer, Linear):
    def __init__(self, name=None, *args, **kwargs):
        super(MPFC, self).__init__(*args, **kwargs)
        self.name = name
        if name is None:
            global FC_LAYER_COUNTER
            FC_LAYER_COUNTER += 1
            self.name = "FC{}".format(FC_LAYER_COUNTER)



    def get_memory(self):
        # returns memory in kbytes
        m_l, m_l_1 = self.weight.size()
        memory = m_l*(m_l_1 + 1) * self.weight_quantizer.get_bits_diff()
        return memory / (1024 * 8)


    def reset_parameters(self):
        keras_replica(self.weight)
        if self.bias is not None:
            init.constant_(self.bias, 0.0)

    def _linear_op(self, x, w, b=None):
        out = torch.nn.functional.linear(x, w, b)
        return out


CONV_LAYER_COUNTER = 0


class MPConv2d(_MPBaseLayer, Conv2d):
    def __init__(self, name=None, *args, **kwargs):
        super(MPConv2d, self).__init__(*args, **kwargs)
        self.name = name
        if name is None:
            global CONV_LAYER_COUNTER
            CONV_LAYER_COUNTER += 1
            self.name = "Conv2D{}".format(CONV_LAYER_COUNTER)




    def reset_parameters(self):
        keras_replica(self.weight)
        if self.bias is not None:
            init.constant_(self.bias, 0.0)

    def _linear_op(self, x, w, b=None):
        out = torch.nn.functional.conv2d(x, w, b, self.stride, (0, 0), self.dilation, self.groups)
        return out

    def get_memory(self):
        # returns memory in kbytes
        m_l, m_l_1, k_l, k_l = self.weight.size()
        memory = m_l*(m_l_1*k_l*k_l + 1) * self.weight_quantizer.get_bits_diff()
        return memory / (1024 * 8)

    def forward(self, x):

        if sum(self.padding) > 0:

            padding = 2 * (self.padding[0],) + 2 * (self.padding[1],)
            x = torch.nn.functional.pad(x, padding, "constant", 0)

        return super(MPConv2d, self).forward(x)


