[toc]

# Distilling the Knowledge in a Neural Network

**keywords:** knowledge distillation

**summary:**

distillation: transfer knowledge from large model to small model

use class probabilities produced by large model as soft targets (e.g. MSE loss between the logits)

$q_i=\frac{\exp(z_i/T)}{\sum_j \exp(z_j/T)}$

$z_i$ is the logits produced by the model, $T$ is the temperature, a high value of $T$ produces a softer distribution

knowledge is transferred to the distilled model by training it on a transfer set and using a soft target distribution for each case in the transfer set that is produced by using the cumbersome model with a high temperature in its softmax