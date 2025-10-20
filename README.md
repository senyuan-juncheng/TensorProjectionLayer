# TensorProjection Layer (TPL)

A tensor-based **dimensionality reduction** layer for deep neural networks.  
Proposed by **Toshinari Morimoto** and **Su-Yun Huang**. Implemented as a custom hidden layer for TensorFlow.

- **Paper (journal)**: Neurocomputing, 2025 (Elsevier) [https://www.sciencedirect.com/science/article/abs/pii/S0925231225023677]
- **Preprint**: arXiv:2004.04454  

## How to use?

The **TensorProjection Layer (TPL)** is typically placed after a 2D convolutional layer.  
The output of a 2D convolutional layer consists of `n` three-dimensional tensors, each of shape `[n, p1, p2, p3]`, where `n` denotes the number of observations (i.e., the minibatch size).

The TPL transforms each input tensor from shape `[n, p1, p2, p3]` to `[n, q1, q2, q3]`,  where each `qk` is less than or equal to the corresponding `pk` for `k = 1, 2, 3`.

If you decide to incorporate the TensorProjection Layer into your model, we suggest placing it after the final convolutional layer for optimal performance.

## Errata (Author Correction)

We have identified errors in the **backpropagation equation** in the published version of the paper.  
These errors **do not affect any of the numerical results or conclusions** presented in the paper.

For details, please see the [Errata and Supplementary Notes](./ERRATA.md).
