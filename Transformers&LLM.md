[toc]

# AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE

**keywords:** Vision Transformer

**summary:**

pre-train on large-scale dataset

<img src="E:\github_repos\paper_reading\fig\VIT.png" alt="VIT" style="zoom: 67%;" />

# Swin Transformer: Hierarchical Vision Transformer using Shifted Windows

**keywords:** Vision Transformer with shifted windows

**summary:**

<img src="E:\github_repos\paper_reading\fig\swin_transformer.png" alt="swin_transformer" style="zoom:67%;" />

 **self-attention in non-overlapping windows:**

 <img src="E:\github_repos\paper_reading\fig\swin_transformer_self_attention.png" alt="swin_transformer_self_attention" style="zoom: 67%;" />

**relative position bias:**

<img src="E:\github_repos\paper_reading\fig\swin_transformer_relative_postition_bias.png" alt="swin_transformer_relative_postition_bias" style="zoom:67%;" />

# End-to-End Object Detection with Transformers

**keywords:** object detection with transformer, set-based global loss with bipartite matching

**summary:**

a fixed size prediction; learned object queries

<img src="E:\github_repos\paper_reading\fig\DETR.png" alt="DETR" style="zoom:67%;" />

<img src="E:\github_repos\paper_reading\fig\DETR2.png" alt="DETR2" style="zoom: 67%;" />

# Improving Language Understanding by Generative Pre-Training

**keywords:** generative pre-training, discriminative finetuning

**summary:**

**unsupervised pre-training**

language modeling objective to maximize the log-likelihood; Decoder only transformer

<img src="E:\github_repos\paper_reading\fig\GPT.png" alt="GPT" style="zoom:50%;" />

**supervised fine-tuning**

include language modeling as auxiliary task in fine-tuning

**task specific transformations**

# BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding

**keywords:** Language Representation Model; BERT

**summary:**

**Task#1 Masked Language Modeling**

**Task#2 Next Sentence Prediction**

<img src="E:\github_repos\paper_reading\fig\BERT.png" alt="BERT" style="zoom:50%;" />

# Training language models to follow instructions with human feedback

**keywords:** LLM, alignment, RLHF

**summary:**

RLHF uses human preferences as reward signals to fine-tune LLM

<img src="E:\github_repos\paper_reading\fig\InstructGPT.png" alt="InstructGPT" style="zoom: 67%;" />

# LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS

**keywords:** low-rank decomposition, fine-tuning

**summary:**

low-rank parametrization update:

$h = W_0x + \Delta Wx = W_0x + BAx$ , 	$W_0 \in \mathbb R^{d\times k} B\in \mathbb R^{d\times r} A\in \mathbb R^{r\times k}$

Gaussian initialization for $A$ and zero initialization for $B$

# LARGE LANGUAGE MODELS AS OPTIMIZERS

**keywords:** gradient-free optimization, prompt optimization, LLM, optimization by prompting

**summary:**

**meta-prompt**

* optimization problem description (meta-instruction)
* optimization trajectory (solution-score pairs sorted in ascending order)

**solution generation**

* optimization stability (generate multiple solutions each time)
* Exploration-exploitation trade-off (tune the temperature)

reach optimum for small-scale problems























































