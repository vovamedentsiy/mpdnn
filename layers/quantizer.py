import torch
import torch.nn as nn
import numpy as np

PARAM_QUANT_COUNTER = 0


class QuantizerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, d, qm, d_hard, qm_hard):
        ctx.save_for_backward(input, d, qm)
        input = torch.clamp(input, min=-qm_hard.item(), max=qm_hard.item())
        output = torch.sign(input) * (d_hard * torch.floor(torch.abs(input) / d_hard  + 1/2)) # d * torch.round(input / d)
        return output

    @staticmethod
    def backward(ctx, grad_output):

        input, d, qm = ctx.saved_tensors
        grad_input = grad_d = grad_qm = None

        input_ = torch.clamp(input, min=-qm.item(), max=qm.item())
        quantized_output = torch.sign(input_) * (d * torch.floor( torch.abs(input_) / d  + 1/2 ) )


        if ctx.needs_input_grad[0]:
            grad_input = grad_output.clone()
            grad_input[torch.abs(input) > qm.item()] = 0 # because I do torch.clamp(input, min=-qm, max=qm) here


        if ctx.needs_input_grad[1]:
            grad_d = (( quantized_output - input) / d) * grad_output
            grad_d[torch.abs(input) > qm.item()] = 0


        if ctx.needs_input_grad[2]:
            grad_qm = torch.sign(input) * grad_output
            grad_qm[torch.abs(input) <= qm.item()] = 0


        return grad_input, grad_d, grad_qm, None, None



class Quantizer(nn.Module):
    def __init__(self, d_init = 2**-3, qm_init = (2**(4-1) - 1)*2**(-3), d_min = 2**-8, d_max = 2**8, qm_min = 2**-8, qm_max = 64, name = None):
        super(Quantizer, self).__init__()

        self.name = name
        if name is None:
            global PARAM_QUANT_COUNTER
            PARAM_QUANT_COUNTER += 1
            self.name = "ParamQuant{}".format(PARAM_QUANT_COUNTER)

        self.integer_bits_constraint = False
        self.parametrization_type = 'LOG'
        if self.parametrization_type == 'LOG':
            self.d_log = nn.Parameter(torch.log(torch.tensor(([d_init]), dtype=torch.float32)), requires_grad=True)
            self.qm_log = nn.Parameter(torch.log(torch.tensor(([qm_init]), dtype=torch.float32)), requires_grad=True)

        elif self.parametrization_type == 'ORIG':
            self.d = nn.Parameter(torch.tensor(([d_init]), dtype=torch.float32), requires_grad=True)
            self.qm = nn.Parameter(torch.tensor(([qm_init]), dtype=torch.float32), requires_grad=True)

        else:
            raise NotImplementedError()

        self.d_init = d_init
        self.qm_init = qm_init
        self.d_min = d_min
        self.d_max = d_max



        self.qm_min = qm_min
        self.qm_max = qm_max



    @torch.no_grad()
    def get_base_grid(self):

        params_dict = self.get_params()
        d, qm = params_dict['d_hard'].item(), params_dict['qm_clamped'].item()
        dict_grid = {}
        for k in range(int(2 ** self.get_bits()) - 1):
            dict_grid[(-qm + k * d).item()] = k

        return dict_grid

    def get_params(self):
        if self.parametrization_type == 'LOG':
            return self.get_params_log()
        elif self.parametrization_type == 'ORIG':
            return self.get_params_orig()


    def get_params_log(self):

        qm_clamped = torch.exp(torch.clamp(self.qm_log, min=np.log(self.qm_min), max=np.log(self.qm_max)))

        d_clamped = torch.exp(torch.clamp(self.d_log, min=np.log(self.d_min), max=np.log(self.d_max)))
        d_clamped = torch.min(d_clamped, qm_clamped)

        d_hard = 2 ** torch.round(torch.log2(d_clamped))
        if self.integer_bits_constraint:
            qm_hard = d_hard * (2 ** (torch.floor(torch.log2(torch.ceil(qm_clamped / d_hard))) + 1) - 1)
        else:
            qm_hard = d_hard * torch.round(qm_clamped / d_hard)
        d_hard = torch.min(d_hard, qm_hard)

        qm_ste = (qm_hard - qm_clamped).detach() + qm_clamped
        d_ste = (d_hard - d_clamped).detach() + d_clamped

        return {'d_clamped': d_clamped, 'qm_clamped': qm_clamped, 'd_hard': d_hard, 'd_ste': d_ste, 'qm_ste': qm_ste,
                'qm_hard': qm_hard}


    def get_params_orig(self):

        qm_clamped = torch.clamp(self.qm, min=self.qm_min, max=self.qm_max)

        d_clamped = torch.clamp(self.d, min=self.d_min, max=self.d_max)
        d_clamped = torch.min(d_clamped, qm_clamped)

        d_hard = 2 ** torch.round(torch.log2(d_clamped))
        if self.integer_bits_constraint:
            qm_hard = d_hard * (2 ** (torch.floor(torch.log2(torch.ceil(qm_clamped / d_hard))) + 1) - 1)
        else:
            qm_hard = d_hard * torch.round(qm_clamped / d_hard)
        d_hard = torch.min(d_hard, qm_hard)

        qm_ste = (qm_hard - qm_clamped).detach() + qm_clamped
        d_ste = (d_hard - d_clamped).detach() + d_clamped


        return {'d_clamped': d_clamped, 'qm_clamped': qm_clamped, 'd_hard': d_hard, 'd_ste': d_ste, 'qm_ste': qm_ste,
                'qm_hard': qm_hard}


    @torch.no_grad()
    def get_bits(self):

        params_dict = self.get_params()
        d_hard, qm_hard = params_dict['d_hard'], params_dict['qm_hard']

        return torch.ceil(torch.log2(qm_hard / d_hard + 1) + 1)


    def get_bits_diff(self):

        params_dict = self.get_params()
        d_ste, qm_ste = params_dict['d_ste'], params_dict['qm_ste']

        bits_hard = torch.ceil(torch.log2(qm_ste / d_ste + 1) + 1)
        bits_smooth = torch.log2(qm_ste / d_ste + 1) + 1

        # I think that bits_smooth should be additionally clamped ( min 2, because then there is an incentive to regularize grid with 2 bits and that should not happen)
        bits_hard = torch.clamp(bits_hard, min=2.)
        bits_smooth = torch.clamp(bits_smooth, min = 2.)

        return (bits_hard - bits_smooth).detach() + bits_smooth


    @torch.no_grad()
    def get_tb_summaries(self):
        tb_summaries = {}
        if self.training:

            params_dict = self.get_params()
            d_ste, qm_ste = params_dict['d_ste'], params_dict['qm_ste']

            if self.parametrization_type == 'LOG':
                tb_summaries[self.name + "/d_log"] = self.d_log.data
                tb_summaries[self.name + "/qm_log"] = self.qm_log.data

            tb_summaries[self.name + "/d"] = d_ste.data
            tb_summaries[self.name + "/qm"] = qm_ste.data
            tb_summaries[self.name + "/bits"] = self.get_bits()

        return tb_summaries


    def forward(self, input):
    
        params_dict = self.get_params()

        return QuantizerFunction.apply(input, params_dict['d_clamped'], params_dict['qm_clamped'], params_dict['d_hard'], params_dict['qm_hard'])

