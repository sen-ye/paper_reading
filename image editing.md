[toc]

# Interactive Point-based Image Editing

## DragDiffusion: Harnessing Diffusion Models for Interactive Point-based Image Editing

**keywords:** diffusion, point-based image editing

**summary:**

<img src="E:\github_repos\paper_reading\fig\DragDiffusion.png" alt="DragDiffusion" style="zoom:50%;" />

**motion supervision:**

<img src="E:\github_repos\paper_reading\fig\DragDiffusion_motion.png" alt="DragDiffusion_motion" style="zoom:40%;" />

**point tracking:**

<img src="E:\github_repos\paper_reading\fig\DragDiffusion_point.png" alt="DragDiffusion_point" style="zoom:40%;" />

## FreeDrag: Point Tracking is Not What You Need for Interactive Point-based Image Editing

**keywords:** point-based image editing, miss tracking, ambiguous tracking, FreeDrag

**summary:**

**issues of DragGAN:** miss tracking, ambiguous tracking

**analysis of DragGAN:**

(1) constant value of handle points' features may lead to failure
(2) desired points lie out of the search area
(3) points disappear
(4) similar features in search area
(5) the search radius $r_2$ is an internal conflict

**adaptive template feature:** to update $F_{ema}$ according to the quality of movement

$F_{ema}^k = \lambda F_r(t_i^k) + (1-\lambda)F_{ema}^{k-1}$

let $L_{ini}^k=\Vert F_{ema}^{k-1}-F_r^{ini}(t_i^k)\Vert_1$ and $L_{end}^k=\Vert F_{ema}^{k-1} - F_r^{end}(t_i^k) \Vert _1$  the expected value of $L_{ini}^{k}$ to be $l$

a larger $l$ denotes a more difficult motion and a smaller $l$ denotes a higher quality of motion, thus the $\lambda$ can be defined as $\lambda  = (1 + \exp (\alpha (L_{end}^k - \beta)))^{-1}$

**fuzzy localization via line search:**

given $F_{ema}^k$ , the next motion is $\mathcal L_{motion} = \Vert F_{ema}^k - F_r(t_i^{k+1}) \Vert _1$

the location of $t_i^{k+1} = S(t_i^k, t_i, F_{ema}^k, d, l)$, where d controls the maximum distance between $t_i^{k+1}$ and $t_i^k$ 





