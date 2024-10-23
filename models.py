import tensorflow as tf
from tensorflow.keras import layers, models, optimizers
from tensorflow.keras.losses import CategoricalCrossentropy
from tensorflow.keras.applications import DenseNet121
from tensor_projection_layer import TensorProjectionLayer
# from tensorflow.keras.regularizers import l2
import tensorflow_addons as tfa


#
# Baseline model
#
def build_baseline_model(image_width, image_height):
    """
    Baseline model: DenseNet121+Conv2D
    """
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))
    base_model.trainable = False
    baseline = models.Sequential([
        base_model,
        layers.Conv2D(64, (1, 1), activation='linear'),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax'),
    ])

    baseline.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=CategoricalCrossentropy(), 
                     metrics=['accuracy',tfa.metrics.F1Score(num_classes=4, average='weighted')])
    
    return baseline

#
# TPL model
#
def build_tpl_model(image_width, image_height):
    """
    TPL model: DenseNet121 + TensorProjectionLayer
    """
    base_model = DenseNet121(weights='imagenet', include_top=False, input_shape=(image_width, image_height, 3))
    base_model.trainable = False
    tpl_model = models.Sequential([
        base_model,
        TensorProjectionLayer(5, 5, 64),
        layers.Flatten(),
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(4, activation='softmax'),
    ])
    
    # focal_loss = CategoricalFocalCrossentropy(gamma=3.0, alpha=0.5)
    tpl_model.compile(optimizer=optimizers.Adam(learning_rate=0.0001), loss=CategoricalCrossentropy(), 
                      metrics=['accuracy',tfa.metrics.F1Score(num_classes=4, average='weighted')])
    
    return tpl_model
