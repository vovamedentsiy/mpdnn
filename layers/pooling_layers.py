
from torch.nn import MaxPool2d, AvgPool2d



MAXPOOL2D_COUNTER = 0


class ProbMaxPooling2d(MaxPool2d):
    def __init__(self, name=None, *args, **kwargs):
        super(ProbMaxPooling2d, self).__init__(*args, **kwargs)
        assert self.kernel_size[0] == self.kernel_size[1]
        if isinstance(self.stride, tuple):
            assert self.stride[0] == self.stride[1]

        if name is None:
            global MAXPOOL2D_COUNTER
            MAXPOOL2D_COUNTER += 1
            name = "MaxPooling2D_{}".format(MAXPOOL2D_COUNTER)
        self.name = name


    def forward(self, x):
        self.input = x
        out = super(ProbMaxPooling2d, self).forward(x)
        self.output = out
        return out


AVPOOL2D_COUNTER = 0


class ProbAvgPool2d(AvgPool2d):
    def __init__(self, name=None, *args, **kwargs):
        super(ProbAvgPool2d, self).__init__(*args, **kwargs)
        if name is None:
            global AVPOOL2D_COUNTER
            AVPOOL2D_COUNTER += 1
            name = "AvgPooling2D_{}".format(AVPOOL2D_COUNTER)
        self.name = name

    def forward(self, x):
        self.input = x
        out = super(ProbAvgPool2d, self).forward(x)
        self.output = out
        return out
