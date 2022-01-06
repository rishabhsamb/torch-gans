# Conditional GAN

This GAN architecture was implemented in a similar fashion to [the original cGAN paper](https://arxiv.org/abs/1411.1784), trained over the MNIST handwritten digit database.

Our generator network is a feedforward network with leaky ReLU activations. Its task is to model the true data distribution and hence generate 'believable' images from the data distribution when provided a random noise input vector. We normalized our images between [-1,1], which is why we chose to use the tanh activation. The generator has an auxiliary input of the class of image we would like it to generate (along with the usual random noise vector). This input is fed into an embedding layer, which after being concatenated with the random noise, is the final input into the generator feedforward network. This is supposed to enforce the 'conditioning' that is presented in the cGAN paper.

Our discriminator network is a 4-layer feedforward network with leaky ReLU activations and a final sigmoid activation as it predicts the probability of the flattened input image being from the true data distribution. Much like the generator, the discriminator also has an auxiliary input of the class of image we would like it to discriminate. Similarly, this input is fed into an embedding layer, which after being concatenated with the original/generated flattened input image, is the final input into the discriminator feedforward network. This is supposed to enforce the 'conditioning' that is presented in the cGAN paper. 

By adding slight perturbations to a few random noise vectors, we can see in the below gif (classes 0-9, in order) how the generator has learnt a function from the noise vector distribution, conditioned on certain input classes to the original data distribution. Qualitatively speaking, we can certainly notice stronger results as compared to the vanilla GAN, as caused by the conditioning on input classes. 

---
### Visualizations

Here we have perturbations to an initial random noise vector. It is _trivial_ what class the generator was conditioned on when generating the image:

![small perturbation gif](https://github.com/rishabhsamb/torch-gans/blob/master/conditional-gan/gifs/0.4_interpolation_cgan.gif)
