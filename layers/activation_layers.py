import torch
import torch.nn.functional as F

from layers.quantizer import Quantizer


QACTIVATION_COUNTER = 0


class QActivation(torch.nn.Module):
    def __init__(self, act_fun=None, quantize_activations = False,name=None, *args, **kwargs):
        super(QActivation, self).__init__(*args, **kwargs)
        assert act_fun in (None,"relu")

        if name is None:
            global QACTIVATION_COUNTER
            QACTIVATION_COUNTER+=1
            name = "QActivation_{}".format(QACTIVATION_COUNTER)
        self.name = name

        self.act_fun = act_fun
        self.return_input = False
        self.tb_summaries_train = {}
        self.tb_summaries_eval = {}

        self.quantize_activations = quantize_activations
        self.quantize_flag = self.quantize_activations

        if self.quantize_activations:
            raise NotImplementedError('Not ready to quantize activations')
            self.activation_quantizer = Quantizer()





    def get_named_parameters(self):
        own_params = {}
        for c in self.children():
            own_params.update({self.name + "/" + k : v for k,v in c.get_named_parameters().items()})
        return own_params




    def _check_input(self,x):
        if self.training:
            pass
        else:
            assert isinstance(x,torch.Tensor)



    def forward(self, x):

        self.input = x
        if self.return_input:
            return x

        if self.quantize_activations:
            out = self.activation_quantizer(x)
        else:
            out = F.relu(x)

        self.output = out

        return out

