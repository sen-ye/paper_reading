[toc]

# Algorithm

## Deep Unsupervised Learning using Non-equilibrium Thermodynamics

**keywords:** deep unsupervised learning, diffusion

**summary:**

the schedule for $\beta_t$ is important in AIS (Annealed Importance Sampling)

<img src="E:\github_repos\paper_reading\fig\diffusion_eq1.png" alt="diffusion_eq1"  />

## Denoising Diffusion Probabilistic Models

**keywords:** latent variable models, denoising diffusion

**summary:**

<img src="E:\github_repos\paper_reading\fig\DDPM.png" alt="DDPM" style="zoom:67%;" />

## Generative Modeling by Estimating Gradients of the Data Distribution

**keywords:** score function, denoising score matching, Langevin dynamics sampling, noise conditional score networks

**summary:**

**manifold assumption**: The manifold hypothesis states that data in the real world tend to concentrate on low dimensional manifolds embedded in a high dimensional space (a.k.a., the ambient space)
**loss function:**

<img src="E:\github_repos\paper_reading\fig\ncsn_loss1.png" alt="ncsn_loss1" style="zoom:67%;" />

<img src="E:\github_repos\paper_reading\fig\ncsn_loss2.png" alt="ncsn_loss2" style="zoom:67%;" />

**sampling:**

<img src="E:\github_repos\paper_reading\fig\ncsn_sampling.png" alt="ncsn_sampling" style="zoom: 50%;" />

**inpainting:**

<img src="E:\github_repos\paper_reading\fig\NCSN_inpainting.png" alt="NCSN_inpainting" style="zoom: 33%;" />

## SCORE-BASED GENERATIVE MODELING THROUGH STOCHASTIC DIFFERENTIAL EQUATIONS

**keywords:** Score-based generation, SDE, ODE

**summary:**

<img src="E:\github_repos\paper_reading\fig\SDE.png" alt="SDE" style="zoom: 33%;" />

**Training:**

<img src="E:\github_repos\paper_reading\fig\SDE_training.png" alt="SDE_training" style="zoom:40%;" />

SDE for SMLD:

<img src="E:\github_repos\paper_reading\fig\SDE_SMLD.png" alt="SDE_SMLD" style="zoom:33%;" />

SDE for DDPM:

<img src="E:\github_repos\paper_reading\fig\SDE_DDPM.png" alt="SDE_DDPM" style="zoom:33%;" />

**probability flow ODE:**

<img src="E:\github_repos\paper_reading\fig\SDE_ODE.png" alt="SDE_ODE" style="zoom:33%;" />

**controllable generation:**

<img src="E:\github_repos\paper_reading\fig\conditional_reverse_SDE.png" alt="conditional_reverse_SDE" style="zoom:33%;" />

## Diffusion Models Beat GANs on Image Synthesis

**keywords:** classifier guidance, architecture improvements

**summary:**

**architecture improvements:**

* Increasing depth versus width, holding model size relatively constant
* Increasing the number of attention heads
* Using attention at 32×32, 16×16, and 8×8 resolutions rather than only at 16×16
* Using the BigGAN residual block for upsampling and downsampling the activations

**classifier guidance:**

<img src="E:\github_repos\paper_reading\fig\classifer_guided_diffusion.png" alt="classifer_guided_diffusion" style="zoom:67%;" />



# Application

## Blended Diffusion for Text-driven Editing of Natural Images

**keywords:** text-driven image editing, CLIP, DDPM

**summary:**

**Local CLIP guided Diffusion:** the cosine distance between local clip image embedding and text embedding+background preserving loss

<img src="E:\github_repos\paper_reading\fig\local_clip_guided_diffusion.png" alt="local_clip_guided_diffusion" style="zoom:50%;" />

<img src="E:\github_repos\paper_reading\fig\local_clip_guided_bgloss.png" alt="local_clip_guided_bgloss" style="zoom:50%;" />

**Text-Driven Blended Diffusion**

<img src="E:\github_repos\paper_reading\fig\text_diriven_blended_diffusion.png" alt="text_diriven_blended_diffusion" style="zoom:50%;" />

## Hierarchical Text-Conditional Image Generation with CLIP Latents

**keywords:** text2image, DALLE2, CLIP, Diffusion

**summary:**

<img src="E:\github_repos\paper_reading\fig\dalle2.png" alt="dalle2" style="zoom:55%;" />

## High-Resolution Image Synthesis with Latent Diffusion Models

**keywords:** conditional image generation, diffusion in latent space, high-resolution

**summary:**

semantic and perceptual compression; 

<img src="E:\github_repos\paper_reading\fig\ldm.png" alt="ldm" style="zoom: 50%;" />
