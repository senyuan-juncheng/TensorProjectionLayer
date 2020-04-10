This README file is under construction.

# TensorProjection Layer 

Thank you for visiting my page :)

TensorProjection Layer is a dimensionality reduction layer proposed by Toshinari Morimoto and Su-Yun Huang.
We implement our proposed method as a custom hidden layer of TensorFlow.

You can put a TensorProjection Layer in a 2D Convolutional Neural Network defined in TensorFlow Keras.
If you would like to run the program, I think that it will be convenient for you to install Python, TensorFlow and Jupyter Notebook via Anaconda. And please run the code (.ipynb file) using Jupyter Notebook.

## Which version of TensorFlow should I use?

Plase do not use a very old version of TensorFlow.
The program needs to compute 'sqrtm' (square root of a positive semi definite matrix), which is not found in an old version of TensorFlow.

## Some Advice

According to our preliminary study, pooling layers are likely to perform better if they are put in the first convolutional layer.
If you wish to use TensorProjection Layer in your model, it may be better for you to put it after the last convolutional layer.

## What does a TensorProjection Layer do?

We assuem that a TensorProjection Layer is installed after a 2D convolutional layer.

The output of a 2D convolutional layer is a tensor with the shape of [n,p1,p2,p3] where n is the number of observations or the minibatch size.

A TensorProjection Layer transforms a tensor with the shape of [n,p1,p2,p3] into one with the shape of [n,q1,q2,q3] where qk is less than or equal to pk (k=1,2,3).

This transformation is done by employing tensor mode product.  The TensorProjection Layer multiplies (trancated) orthogonal matrices with the size of [qk,pk] (k=1,2,3) to each 3D tensor (i=1,..,n).

## Our Related Method

We also have a matrix version of Projection Layer. We will upload it in the near future.
