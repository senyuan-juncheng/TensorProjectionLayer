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
