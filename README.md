# Mixed Precision DNNs: All you need is a good parametrization
Pytorch implementation of the [Differentiable Quantization](https://arxiv.org/abs/1905.11452) 

In this repository, you can find the implementation of the DQ uniform quantizer with the 3-rd type of parametrization (U3), the one which yields the best results among other parametrization variants. 

The quantizer is implemented for the weights, and could be easily extended for activations with minor changes in code. 

We train Resnet20 on Cifar10 to illustrate results of quantization. 
