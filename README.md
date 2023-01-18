# Mask Image Modeling on Self-Supervision
This project is part of my MSc Thesis https://pergamos.lib.uoa.gr/uoa/dl/frontend/el/browse/3256221  for my postgraduate studies in Data Science and Information Technologies (NKUA).

In this project, we propose new masking strategies that achieve higher k-NN, linear probing scores and acceleration in the learning process of downstream tasks. Considering the computational efficiency challenge these methods face, we conduct experiments on different scales of a dataset and number of training epochs and show their impact on the scores. Finally, we introduce a new loss function based on *contrastive learning* and achieve improvements over the baseline when used with different masking strategies.


## Baseline approaches

- [iBOT](https://arxiv.org/abs/2111.07832)
- [AttMask](https://arxiv.org/abs/2203.12719)

## Proposed Masking Strategies

- Mask generation from different layers raw attention maps 

- [Rollout](https://arxiv.org/abs/2005.00928) method for mask generation

- Pre-processing of attention maps, with log and power functions, for competitive MIM

- Multi-layer mask generation 

To understand better the way each Mask is generated, we suggest taking a look at the [Thesis](https://pergamos.lib.uoa.gr/uoa/dl/frontend/el/browse/3256221) on page 58.
  
## Dataset 

- We conduct all the experiments on ImageNet dataset. During training we use the full train set of Imagenet or a subset (the first 20% of training samples per class).
- For the evaluation of the models, we use the full validation set of ImageNet.






## Clone the repo

To clone the repo:

```bash
  git clone https://github.com/DimitrisReppas/MIM_on_self-supervision.git
```

## Requirements

The requirements of the project can be found [here](https://github.com/DimitrisReppas/MIM_on_self-supervision/blob/main/requirements.txt). 

## Implementation details 

- HPC resources were used from GENCI-IDRIS (Grant 2020-AD011013552).

- The code is designed to perform distributed parallel processing. 

- All the implementation details can be found: [Thesis](https://pergamos.lib.uoa.gr/uoa/dl/frontend/el/browse/3256221) on page 61

## Training of the models 

To conduct a training experiment:

```bash
  sbatch train_example.slurm
```
Try the proposed Masking Strategies, by changing the following arguments:

`--mask` `--layer_mask` `--power`

## Contrastive Learning

To train the models with the proposed contrastive term, replace `multimask_main_ibot.py` with `contrastive_3rd_term_main_ibot.py` in `train_example.slurm`.

All the hyperparameters are found: [Thesis](https://pergamos.lib.uoa.gr/uoa/dl/frontend/el/browse/3256221) on page 61 

## Evaluation

### k-NN

Use k-NN to evaluate the models:

```bash
  sbatch k-nn_example.slurm
```

### Linear probing

Use linear probing to evaluate the models:

```bash
  sbatch linear_probing_example.slurm 
```

## Results

Bellow, some results of the project are presented:

- Linear probing and k-NN scores of AttMask for different layers (trained on the subset dataset)

![fig_1](https://user-images.githubusercontent.com/74918841/213267431-8eaa78ff-9680-415b-b3b5-6181f33508a5.png)

- Evaluation of Rollout-based masking strategies with k-NN and linear probing (trained on the subset dataset)

![fig_2](https://user-images.githubusercontent.com/74918841/213267957-63deda7b-e0b6-4d8a-af6d-7b5cbbe426c7.png)

- Linear probing and k-NN evaluation of masking strategies based on the pre-processing of the attention maps with power and log functions (trained on the subset dataset)

![fig_3](https://user-images.githubusercontent.com/74918841/213268197-7386de56-d449-4907-99de-0e77f20d60e9.png)

- Evaluation of the multi-layer masking and multi-crop strategies with k-NN and linear probing (trained on the subset dataset)

![fig_4](https://user-images.githubusercontent.com/74918841/213268507-f5ed0bcc-17d8-4191-8b28-d71fac50c8ac.png)

- Evaluation of masking strategies with k-NN and linear probing (trained on full ImageNet)

![fig_5](https://user-images.githubusercontent.com/74918841/213268939-e9d04777-6a3c-4db8-b147-1e1dc3061f6b.png)

- A closer look at linear probing plots

![fig_6](https://user-images.githubusercontent.com/74918841/213269338-26ba5857-ee40-4259-9108-9c315d7e2f96.png)

- Contrastive Learning results (trained on the subset dataset)

![fig_7](https://user-images.githubusercontent.com/74918841/213269859-a1dba54b-232e-4f9d-8f9a-babf5268a145.png)

## Acknowledgement

This repository is built using the [iBOT](https://github.com/bytedance/ibot) repository and inspired by [AttMask](https://arxiv.org/abs/2203.12719) paper.
