# Conditional GAN MNIST

A Conditional Generative Adversarial Network (CGAN) implemented 
using convolution layers to produce synthetic images of hand drawn digits
(Implemented using Tensoflow 2).

## 1. Background

### 1.1 Vanilla Generative Adversarial Network
The Generative Adversarial Network (GAN) is a state-of-the-art
technique for estimating generative models proposed by Goodfellow et al [1].
The framework consists of two models: 
- Generator: Captures the distribution of the data and produces synthetic examples
based on a sample from a latent distribution (Gaussian in this case).
- Discriminator: Estimates the probability that a training example is from the training 
dataset rather than being produced by the Generator.

Through the training process, the generator is optimised to produce
better samples to lower the Discriminator's certainty that a sample is fake.

### 1.2 Deep Convolutional GAN

Radford et al. extended the proposed GAN framework to use deep
convolutional layers [2]. They proposed a number of recommended
guidelines for the architecture to improve optimisation:

- Replace any pooling layers with strided convolutions (Discriminator) and fractional-strided
convolutions (Generator).
- Use batchnorm in both the generator and the discriminator.
- Remove fully connected hidden layers for deeper architectures.
- Use ReLU activation in generator for all layers except for the output, which uses Tanh.
- Use LeakyReLU activation in the discriminator for all layers.

In this project, all of these guidelines have been followed with the exception
of LeakyReLU being used in the generator instead of the recommended ReLU as this
was found to work better.

### 1.3 Conditional GANs

The traditional GAN architectures produce synthetic examples
exclusively on a sample from the latent space. In many cases,
it is useful to add some conditioned information `y` to deliberately 
produce synthetic examples of a certain type. In this case, the 
conditioned information is the digit that we want to produce a 
hand-drawn image of. Mirza et al. proposed a method of incorperating
the conditioned information `y` on the MNIST using a Vanilla GAN [3].

## 2. Install

Create Environment: `conda create --name cgan python=3.6`

Activate Environment: `conda activate cgan`

Install Requirements: `pip install -r requirements.txt`

## 3. Run

Generate: `python -m examples.example_generate`

Train: `python -m examples.example_train`

## 4. Generated Examples

A number of example generated images can be found in the
`generated_images` directory.

## 5. References

[1] Goodfellow, Ian, et al. "Generative adversarial nets." Advances in neural information processing systems. 2014.
https://arxiv.org/pdf/1406.2661.pdf

[2] Radford, Alec, Luke Metz, and Soumith Chintala. "Unsupervised representation learning with deep convolutional generative adversarial networks." arXiv preprint arXiv:1511.06434 (2015).
https://arxiv.org/pdf/1511.06434.pdf

[3] Mirza, Mehdi, and Simon Osindero. "Conditional generative adversarial nets." arXiv preprint arXiv:1411.1784 (2014).
https://arxiv.org/pdf/1411.1784.pdf