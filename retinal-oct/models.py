import tensorflow as tf
from tensorflow.keras import layers, models
from tensor_projection_layer import TensorProjectionLayer

#
# base model
#

def create_base_model(image_width, image_height):
    base_model = models.Sequential([
    # 1st conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_width, image_height, 3)),
    layers.MaxPooling2D((2, 2)),
    # 2nd conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # 3rd conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # 4th conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # flatten and dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(4, activation='softmax')])

    base_model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return base_model

#
# TPL 1st
#
def create_tpl_1st(image_width, image_height):
    tpl_1st = models.Sequential([
    # 1st conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_width, image_height, 3)),
    layers.MaxPooling2D((2, 2)),
    # 2nd conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # 3rd conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # 4th conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    TensorProjectionLayer(6,6,32),
    # flatten and dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(4, activation='softmax')])

    tpl_1st.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return tpl_1st

#
# TPL 2nd
#
def create_tpl_2nd(image_width, image_height):
    tpl_2nd = tpl_2nd = models.Sequential([
    # 1st conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(image_width, image_height, 3)),
    layers.MaxPooling2D((2, 2)),
    # 2nd conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # 3rd conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2)),
    # 4th conv
    layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
    TensorProjectionLayer(6,6,20),
    # flatten and dense layers
    layers.Flatten(),
    layers.Dense(256, activation='relu'),
    layers.Dense(4, activation='softmax')
])

    tpl_2nd.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    
    return tpl_2nd
