[toc]

# Autoregressive Models

## MADE: Masked Autoencoder for Distribution Estimation

**keywords:** autoencoder, density estimation, mask, autoregressive

**Summary:** masked autoencoder for high-dimensional discrete data distribution estimation with a single pass

## Pixel Recurrent Neural Networks

**keywords:** autoregressive, image generation, masked convolution

**summary:** PixelRNN and PixelCNN

## PixelSNAIL: An Improved Autoregressive Generative Model

**keywords:** causal convolution, masked attention, improved PixelCNN

**summary:** The authors combine self-attention with causal convolutions to model long-range dependencies in neural autoregressive models. There are two blocks in PixelSNAIL: residual block and attention block. Empirical experiments validate PixelSNAIL achieves a larger receptive field than previous models.

## Conditional Image Generation with PixelCNN Decoders

**keywords:** conditional image generation, Gated PixelCNN

**summary:** several improvements to original PixelCNN: Gated PixelCNN to improve model's capacity; fix blind spot; conditional generation by conditioning on a latent $h$ 

location independent generation: 
<img src="E:\github_repos\paper_reading\fig\f1.png" alt="f1" style="zoom: 67%;" />
location dependent generation:
<img src="E:\github_repos\paper_reading\fig\location_dependent_gen.png" alt="location_dependent_gen" style="zoom:67%;" />

# Flow Model

## DENSITY ESTIMATION USING REAL NVP

**keywords:** real NVP, normalizing flow, exact likelihood, fast sampling, latent space

**summary:**
change of variable formula:

<img src="E:\github_repos\paper_reading\fig\change_of_variable.png" alt="change_of_variable" style="zoom:67%;" />

the determinant of the Jacobian Matrix is normally hard to compute but a determinant of a triangular can be easily computed

**coupling layer**:

<img src="E:\github_repos\paper_reading\fig\coupling_layer.png" alt="coupling_layer" style="zoom:67%;" />

invertible function; determinant easy to compute; composite coupling layers to achieve complex transformations

## Glow: Generative Flow with Invertible 1×1 Convolutions

**keywords:** Flow, 1x1 invertible convolution

**summary:**

Glow model components:

<img src="E:\github_repos\paper_reading\fig\glow.png" alt="glow" style="zoom:67%;" />



# Latent Variable Model

## Auto-Encoding Variational Bayes

**keywords:** variational lower bound, Stochastic Gradient Variational Bayes

**summary:** encoder parameterized by $\phi$ to approximate the posterior distribution, decoder parameterized by $\theta$ to model the generation process; variational lower bound for $\log p_\theta (x)$ is the negative KL divergence between approximate posterior distribution $q_\phi (z|x)$  and a prior distribution $p_\theta(z)$ plus a reconstruction loss term $E_{q_{\phi}(z|x)}[\log p_\theta(x|z)]$; re-parameterization trick for optimizing the reconstruction loss term

## IMPORTANCE WEIGHTED AUTOENCODERS

**keywords:** importance sampling, autoencoder, VAE, IWAE

**summary:** a tighter lower bound on $\log p_\theta(x)$ using importance weighting; the proposed objective function can learn richer hidden representations than vanilla VAE; measure active units (effective dimensions) in hidden representations  

## Neural Discrete Representation Learning

**keywords:** VQ-VAE, discrete latent representations

**summary:** discrete latent space; use PixelCNN to learn the prior sample; input passed through encoder then mapped to nearest latent embedding and fed into decoder; loss=reconstruction+VQloss+commitment

<img src="E:\github_repos\paper_reading\fig\vqvae_loss.png" alt="vqvae_loss" style="zoom:67%;" />

## Generating Diverse High-Fidelity Images with VQ-VAE-2

**keywords:** VQ-VAE-2, hierarchical discrete latent representations, PixelCNN

**summary:** generate high resolution images by hierarchical latent representations; use PixelSNAIL and PixelCNN to model hierarchical latent distribution

<img src="E:\github_repos\paper_reading\fig\vqvae2.png" alt="vqvae2" style="zoom:67%;" />

# Implicit Model

## Generative Adversarial Nets

**keywords:** implicit density model, GAN

**summary:** a minimax game between two players

<img src="E:\github_repos\paper_reading\fig\GANLoss.png" alt="GANLoss" style="zoom:67%;" />

the game achieves optimum when $p_g=p_{data}$ 

<img src="E:\github_repos\paper_reading\fig\GANoptimum.png" alt="GANoptimum" style="zoom:67%;" />

## UNSUPERVISED REPRESENTATION LEARNING WITH DEEP CONVOLUTIONAL GENERATIVE ADVERSARIAL NETWORKS

**keywords:** deep convolutional GAN

**summary:**

Techniques to stabilize training of DCGAN: (1) replace pooling layer with strided convolution (2) Batchnorm (3) remove fully connected layer (4) ReLU in generator and Leaky ReLU in discriminator

use discriminator to do downstream image classification task

## Improved Techniques for Training GANs

**keywords:** GAN, stabilize training of GAN

**summary:**

**Feature Matching:** train the generator to match the expected value of the features on an intermediate layer of the discriminator

**Minibatch discrimination:** allow the discriminator to look at multiple data examples in combination

<img src="E:\github_repos\paper_reading\fig\minibatch_discrimination.png" alt="minibatch_discrimination" style="zoom:50%;" />

**Historical Averaging of model's parameters** 

**one-sided label smoothing:** only smooth the positive labels to .9 for example

**virtual batch normalization:** each example is normalized based on  statistics of a reference batch

**Inception Score:** $\exp(E_x \mathbf{KL}(p(y|x),p(y)))$

**Semi-supervised learning:** adding supervised learning objective to classify labels and this improves the quality of samples

## PROGRESSIVE GROWING OF GANS FOR IMPROVED QUALITY, STABILITY, AND VARIATION

**keywords:** ProGAN

**summary:**

**(1) increase variation by minibatch standard deviation**

**(2) Equalized learning rate**

**(3) Pixel-wise feature vector normalization:** normalize the feature vector in each pixel to unit length in the generator after each convolutional layer

<img src="E:\github_repos\paper_reading\fig\ProGAN.png" alt="ProGAN" style="zoom:50%;" />

## Self-Attention Generative Adversarial Networks

**keywords:** SAGAN

**summary:**

<img src="E:\github_repos\paper_reading\fig\SAGAN.png" alt="SAGAN" style="zoom:55%;" />

**learnable residual connection:** $o_i$ is the output from self-attention layer and $\gamma$ is the learnable parameter initialized to 0

<img src="E:\github_repos\paper_reading\fig\SAGAN_aggregation.png" alt="SAGAN_aggregation" style="zoom: 67%;" />

**hinge loss:** hinge version of adversarial loss

<img src="E:\github_repos\paper_reading\fig\SAGAN_loss.png" alt="SAGAN_loss" style="zoom:50%;" />

## A Style-Based Generator Architecture for Generative Adversarial Networks

**keywords:** StyleGAN, Style transfer, disentangle, control of image synthesis

**summary:**

style control by AdaIN; the mean and variance of a feature map controls the style

<img src="E:\github_repos\paper_reading\fig\StyleGAN.png" alt="StyleGAN" style="zoom:50%;" />

## Notes On GANs, Energy-Based Models, and Saddle Points

**keywords:** GAN, Energy-based model, saddle point

**summary:**

Minimization of negative log likelihood of EBM:

<img src="E:\github_repos\paper_reading\fig\EBM_ML.png" alt="EBM_ML" style="zoom:50%;" />

**minimax problems** $min_x max_y f(x, y)$

**saddle point** ($x*, y*$) at which we can change the min and max

**convex-concave** $f(x,y)$ is convex in x and concave in y

**Theorem:** if f is convex-concave then we can always change the min-max and for any saddle we have $\nabla_x f(x,y)=\nabla_y f(x,y) = 0$  

## Drag Your GAN: Interactive Point-based Manipulation on the Generative Image Manifold

**keywords:** GAN, point based image editing, point tracking

**summary:**

input: handle points, target points

<img src="E:\github_repos\paper_reading\fig\DragGAN.png" alt="DragGAN" style="zoom:50%;" />

(1) motion supervision

<img src="E:\github_repos\paper_reading\fig\DragGAN_motion_supervision.png" alt="DragGAN_motion_supervision" style="zoom:50%;" />

<img src="E:\github_repos\paper_reading\fig\DragGAN_motion_loss.png" alt="DragGAN_motion_loss" style="zoom: 50%;" />

(2) point tracking

<img src="E:\github_repos\paper_reading\fig\DragGAN_point_tracking.png" alt="DragGAN_point_tracking" style="zoom: 50%;" />

# Self-supervised Learning

## Context Encoders: Feature Learning by Inpainting

**keywords:** unsupervised visual feature learning, context-based pixel prediction 

**summary:**

<img src="E:\github_repos\paper_reading\fig\context-encoder.png" alt="context-encoder" style="zoom:67%;" />

**masking strategy:**

<img src="E:\github_repos\paper_reading\fig\context-encoder-mask.png" alt="context-encoder-mask" style="zoom: 50%;" />

**adversarial loss:**

<img src="E:\github_repos\paper_reading\fig\contex-encoder-advloss.png" alt="contex-encoder-advloss" style="zoom:50%;" />

## Split-Brain Autoencoders: Unsupervised Learning by Cross-Channel Prediction

**keywords:**  cross-channel auto-encoder, unsupervised pre-training

**summary:**

pre-trained on full data tensor; no need for bottleneck

<img src="E:\github_repos\paper_reading\fig\split-brain-auto-encoder.png" alt="split-brain-auto-encoder" style="zoom:50%;" />

## Tracking Emerges by Colorizing Videos

**keywords:** colorization, self-supervised learning, tracking, video

**summary:**

copy color from a reference frame; colorization causes tracking to emerge; 

compute similarity matrix between the target and reference frame (on a low-resolution image):

<img src="E:\github_repos\paper_reading\fig\tracking_emerge_coloring_video1.png" alt="tracking_emerge_coloring_video1" style="zoom:50%;" />

the predicted color $y_j=\sum_i A_{ij}c_i$, the loss function is cross-entropy loss

## Efficient Estimation of Word Representations in Vector Space

**keywords:** representation of words, word2vec, Neural Net Language Model

**summary:**

<img src="E:\github_repos\paper_reading\fig\cbow_skip_gram.png" alt="cbow_skip_gram" style="zoom:50%;" />

## Representation Learning with Contrastive Predictive Coding

**keywords:** Contrastive Predictive Coding, Mutual Information, InfoNCE loss

**summary:**

InfoNCE Loss:

<img src="E:\github_repos\paper_reading\fig\infonce.png" alt="infonce" style="zoom:50%;" />

maximize the mutual information=minimize the InfoNCE loss
classify future representations among a set of unrelated "negative" representations

## Momentum Contrast for Unsupervised Visual Representation Learning

**keywords:** contrastive learning, momentum encoder, unsupervised

**summary:**

three kinds of contrastive loss design:

<img src="E:\github_repos\paper_reading\fig\moco1.png" alt="moco1" style="zoom: 67%;" />

the key encoder is updated by momentum: $\theta_k=m\theta_k+(1-m)\theta_q$, where m is typically 0.999, to maintain a consistent key encoder

goal: large negative samples, consistent key representations

## A Simple Framework for Contrastive Learning of Visual Representations

**keywords:** unsupervised, contrastive 

**summary:**

**good contrastive learning:** (1) multiple data augmentations (2) learnable non-linear transformation between represetation and contrastive loss (3) larger batch size and longer training 

<img src="E:\github_repos\paper_reading\fig\SimCLR.png" alt="SimCLR" style="zoom:67%;" />

## Learning Transferable Visual Models From Natural Language Supervision

**keywords:** CLIP, contrastive, supervision from Natural Language

**summary:** 

<img src="E:\github_repos\paper_reading\fig\CLIP.png" alt="CLIP" style="zoom:67%;" />

## Masked Autoencoders Are Scalable Vision Learners

**keywords:** MAE, ViT

**summary:**

Encoder: input: 25% of the image tokens (no mask tokens)
Decoder: encoded tokens and [MASK] tokens
Loss: reconstruction MSE loss only on the masked tokens

<img src="E:\github_repos\paper_reading\fig\MAE.png" alt="MAE" style="zoom:50%;" />

# Semi-supervised Learning

## Pseudo-Label : The Simple and Efficient Semi-Supervised Learning Method for Deep Neural Networks

**keywords:** semi-supervised learning, pseudo-label, entropy regularization

**summary:**

For unlabeled data, Pseudo-Labels, just picking up the class which has the maximum predicted probability every weights update, are used as if they were true labels

**cluster assumption:** the decision boundary should lie in low-density regions to improve generalization performance

pseudo-label is equivalent to entropy regularization

## TEMPORAL ENSEMBLING FOR SEMI-SUPERVISED LEARNING

**keywords:** semi-supervised learning, self-ensembling

**summary:**

$\Pi$-Model encourages consistent network output between two realizations of the same input stimulus under two different dropout conditions.

Temporal ensembling simplifies and extends this by taking into account the network predictions over multiple previous training epochs.

<img src="E:\github_repos\paper_reading\fig\SSL_temporal_ensembling.png" alt="SSL_temporal_ensembling" style="zoom:50%;" />

## Mean teachers are better role models: Weight-averaged consistency targets improve semi-supervised deep learning results

**keywords:** Mean teacher, semi-supervised learning

**summary:**

Mean teacher: a method that average model weights

<img src="E:\github_repos\paper_reading\fig\mean_teacher.png" alt="mean_teacher" style="zoom: 67%;" />

consistency cost: student model ($\theta \space \eta$); teacher model($\theta ' \space \eta '$)

<img src="E:\github_repos\paper_reading\fig\mean_teacher_consistency_cost.png" alt="mean_teacher_consistency_cost" style="zoom:50%;" />

## Self-training with Noisy Student improves ImageNet classification

**keywords:** semi-supervised learning, noisy student, self-training

**summary:**

self-training framework:

* train a teacher model on labeled images
* use the teacher to generate pseudo labels on unlabeled images (teacher should not be noised)
* train a student model on the combination of labeled images and pseudo labeled images (student should be noised is the key)

<img src="E:\github_repos\paper_reading\fig\NoisyStudent.png" alt="NoisyStudent" style="zoom:50%;" />























