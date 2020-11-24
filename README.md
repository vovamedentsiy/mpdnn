# Mixed Precision DNNs: All you need is a good parametrization
Pytorch implementation of the [Differentiable Quantization](https://arxiv.org/abs/1905.11452) 

In this repository, you can find the implementation of the DQ uniform [quantizer with the 3-rd type of parametrization (U3)](https://github.com/vovamedentsiy/mpdnn/blob/main/layers/quantizer.py), the one which yields the best results among other parametrization variants. 

The quantizer is implemented for the weights, and could be easily extended for activations with minor changes in code. 

We train Resnet20 on Cifar10 to illustrate results of quantization. 


<p> 
    <img src="https://github.com/vovamedentsiy/mpdnn/blob/main/imgs/Validation_Accuracy.svg" width="400" height ="250" />
    <br>
    <em>ELBO for VAE with 20-dim latent space</em> 
<p\>
