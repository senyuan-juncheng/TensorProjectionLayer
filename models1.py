#
# models1:
# A standard U-NET architecture
# Input must be 16n x 16n
#

import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensor_projection_layer import TensorProjectionLayer, TransposeTensorProjectionLayer

#
# tpl unet (the 4th MaxPooling --> TPL; the 1st transposeConvd2d -->TPL)
#
def tpl_unet_model(input_size=(128, 128, 3), c=32, leaky_relu=False):
    inputs = layers.Input(input_size)
    
    # the size of TensorProjectionLayer
    tpl_size_h = input_size[0] // 16  # height
    tpl_size_w = input_size[1] // 16  # width

    # Encoder
    c1 = layers.Conv2D(c, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(c, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(c*2, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(c*2, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(c*4, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(c*4, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(c4)
    p4 = TensorProjectionLayer(tpl_size_h, tpl_size_w, c*8)(c4)  # (tpl_size_h, tpl_size_w, c*8)

    # if True apply leaky_relu activation
    if leaky_relu:
        p4 = layers.LeakyReLU(alpha=0.1)(p4)

    # Bottleneck
    c5 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u4 = TransposeTensorProjectionLayer(tpl_size_h*2, tpl_size_w*2, c*8)(c5)  # (tpl_size_h*2, tpl_size_w*2, c*8)
    u4 = layers.concatenate([u4, c4])
    c6 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(u4)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    
    u3 = layers.Conv2DTranspose(c*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u3 = layers.concatenate([u3, c3])
    c7 = layers.Conv2D(c*4, (3, 3), activation='relu', padding='same')(u3)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(c*4, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    
    u2 = layers.Conv2DTranspose(c*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u2 = layers.concatenate([u2, c2])
    c8 = layers.Conv2D(c*2, (3, 3), activation='relu', padding='same')(u2)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(c*2, (3, 3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    
    u1 = layers.Conv2DTranspose(c, (2, 2), strides=(2, 2), padding='same')(c8)
    u1 = layers.concatenate([u1, c1])
    c9 = layers.Conv2D(c, (3, 3), activation='relu', padding='same')(u1)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(c, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    
    # output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    
    return model


#
# unet (baseline)
#
def unet_model(input_size=(128, 128, 3), c=32):
    inputs = layers.Input(input_size)
    
    # Encoder
    c1 = layers.Conv2D(c, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(c, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)
    
    c2 = layers.Conv2D(c*2, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(c*2, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(c*4, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(c*4, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)
    
    c4 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(c4)
    p4 = layers.MaxPooling2D((2, 2))(c4)

    # Bottleneck
    c5 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(p4)
    c5 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(c5)
    
    # Decoder
    u4 = layers.Conv2DTranspose(c*8, (2, 2), strides=(2, 2), padding='same')(c5)
    u4 = layers.concatenate([u4, c4])
    c6 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(u4)
    c6 = layers.BatchNormalization()(c6)
    c6 = layers.Conv2D(c*8, (3, 3), activation='relu', padding='same')(c6)
    c6 = layers.BatchNormalization()(c6)
    
    u3 = layers.Conv2DTranspose(c*4, (2, 2), strides=(2, 2), padding='same')(c6)
    u3 = layers.concatenate([u3, c3])
    c7 = layers.Conv2D(c*4, (3, 3), activation='relu', padding='same')(u3)
    c7 = layers.BatchNormalization()(c7)
    c7 = layers.Conv2D(c*4, (3, 3), activation='relu', padding='same')(c7)
    c7 = layers.BatchNormalization()(c7)
    
    u2 = layers.Conv2DTranspose(c*2, (2, 2), strides=(2, 2), padding='same')(c7)
    u2 = layers.concatenate([u2, c2])
    c8 = layers.Conv2D(c*2, (3, 3), activation='relu', padding='same')(u2)
    c8 = layers.BatchNormalization()(c8)
    c8 = layers.Conv2D(c*2, (3, 3), activation='relu', padding='same')(c8)
    c8 = layers.BatchNormalization()(c8)
    
    u1 = layers.Conv2DTranspose(c, (2, 2), strides=(2, 2), padding='same')(c8)
    u1 = layers.concatenate([u1, c1])
    c9 = layers.Conv2D(c, (3, 3), activation='relu', padding='same')(u1)
    c9 = layers.BatchNormalization()(c9)
    c9 = layers.Conv2D(c, (3, 3), activation='relu', padding='same')(c9)
    c9 = layers.BatchNormalization()(c9)
    
    # output
    outputs = layers.Conv2D(1, (1, 1), activation='sigmoid')(c9)
    
    model = models.Model(inputs=[inputs], outputs=[outputs])
    
    return model
