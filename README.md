# Forward-Only Convolutional Neural Networks with Learnable Channel-Class Assignment (CAW-Conv)

> Mohammadnavid Ghader, Saeed Reza Kheradpisheh, Bahar Farahani, Mahmood Fazlali
---

## Overview

The Forward-Forward (FF) algorithm has emerged as a biologically inspired alternative to backpropagation by replacing global gradient propagation with local forward-only learning objectives.

This repository introduces **Class-Adaptive Weighted Convolution (CAW-Conv)**, a novel forward-only convolutional learning framework that improves feature specialization and channel utilization through:

* **Learnable Channel-Class Assignment**
* **Entropy Regularization**
* **Orthogonality Regularization**
* **Loss-Aware Layer Contribution Strategy**
* **Fully Local Layer-Wise Optimization**
* **Deep Residual Forward-Only CNN Training**

Unlike previous FF-based convolutional methods that rely on static channel grouping, CAW-Conv dynamically learns how each convolutional channel contributes to different classes during training.

---

## Main Results

### CIFAR-10, MNIST, Fashion-MNIST

| Method              | Architecture  | CIFAR-10  | MNIST     | Fashion-MNIST |
| ------------------- | ------------- | --------- | --------- | ------------- |
| FF                  | MLP           | 59.00     | 98.69     | -             |
| SymBa               | MLP           | 59.09     | 98.58     | -             |
| CaFo                | CNN           | 67.43     | 98.80     | -             |
| CwComp              | CNN           | 78.11     | 99.42     | 92.31         |
| DeeperForward       | CNN           | 86.22     | 99.63     | 93.13         |
| **CAW-Conv (Ours)** | **ResNet-17** | **89.37** | **99.74** | **94.55**     |

### CIFAR-100 and Tiny-ImageNet

| Method               | CIFAR-100 | Tiny-ImageNet |
| -------------------- | --------- | ------------- |
| DeeperForward        | 53.09     | 41.36         |
| DeeperForward (CH×3) | 60.28     | -             |
| **CAW-Conv**         | **63.52** | **49.87**     |
| **CAW-Conv (CH×3)**  | **69.74** | -             |

---

## Publication Details

- **arXiv:** https://arxiv.org/abs/2606.09928
- **PDF:** https://arxiv.org/pdf/2606.09928.pdf

---

## Citation

If you find this work useful, please cite:

```bibtex
@article{ghader2026forwardonly,
  title={Forward-Only Convolutional Neural Networks with Learnable Channel-Class Assignment},
  author={Ghader, Mohammadnavid and Kheradpisheh, Saeed Reza and Farahani, Bahar and Fazlali, Mahmood},
  journal={arXiv preprint arXiv:2606.09928},
  year={2026}
}
```


