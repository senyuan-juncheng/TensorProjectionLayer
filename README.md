# TensorProjection Layer 

Thank you for visiting my page :)

The TensorProjection Layer is a dimensionality reduction layer proposed by Toshinari Morimoto and Su-Yun Huang. 
We have implemented our method as a custom hidden layer in TensorFlow.

You can integrate a TensorProjection Layer into a Deep Neural Network defined using TensorFlow Keras. 
To get started, we recommend installing Python, TensorFlow, and Jupyter Notebook.
You can then run the provided code (.ipynb file) using Jupyter Notebook.

## TensorFlow Version Requirements

Please avoid using very old versions of TensorFlow. The TensorProjection Layer requires the computation of the square root of a positive semi-definite matrix (`sqrtm`), which is not available in older versions of TensorFlow. Ensure that your version supports this function.

## Practical Advice

If you decide to incorporate the TensorProjection Layer into your model, we suggest placing it after the final convolutional layer for optimal performance.

## What is the TensorProjection Layer?

The TensorProjection Layer is typically installed after a 2D convolutional layer. 
The output of a 2D convolutional layer consists of `n` 3D tensors, each shaped as [n, p1, p2, p3], where `n` is the number of observations (i.e., the minibatch size).

The TensorProjection Layer transforms the input tensor from a shape of [n, p1, p2, p3] to [n, q1, q2, q3], where each `qk` is less than or equal to `pk` (for k=1, 2, 3).

This transformation is achieved by performing a tensor mode product on each 3D tensor (i=1,2,...,n). The TensorProjection Layer multiplies truncated orthogonal matrices of size [qk, pk] (for k=1, 2, 3) with each tensor.

## Paper

For further details, please refer to our paper:  
[https://arxiv.org/abs/2004.04454](https://arxiv.org/abs/2004.04454)
