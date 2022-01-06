# Conditional GAN

This GAN architecture was implemented in a similar fashion to [the original GAN paper](https://arxiv.org/abs/1406.2661), trained over the MNIST handwritten digit database.

Our generator network is a feedforward network with leaky ReLU activations. Its task is to model the true data distribution and hence generate 'believable' images from the data distribution when provided a random noise input vector. We normalized our images between [-1,1], which is why we chose to use the tanh activation. 

Our discriminator network is a 4-layer feedforward network with leaky ReLU activations and a final sigmoid activation as it predicts the probability of the flattened input image being from the true data distribution.

By adding slight perturbations to a few random noise vectors, we can see how the generator has learnt a function from the noise vector distribution to the original data distribution.

---
### Visualizations

Here we have small perturbations to an initial random noise vector:

![small perturbation gif](https://github.com/rishabhsamb/torch-gans/blob/master/vanilla-gan/gifs/0.1%20interpolation.gif)


Here we have larger perturbations to another random noise vector:

![large perturbation gif](https://github.com/rishabhsamb/torch-gans/blob/master/vanilla-gan/gifs/0.5%20interpolation.gif)
