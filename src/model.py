"""
Model building utilities: transfer learning scaffold and a small custom CNN.
"""

import tensorflow as tf
from tensorflow.keras import layers, models


def build_transfer_model(backbone_name='ResNet50', input_shape=(224, 224, 3), dropout=0.5):
    """Builds a transfer learning model with a frozen backbone by default.

    Args:
        backbone_name (str): 'ResNet50' | 'DenseNet121' | 'VGG16'
        input_shape (tuple)
        dropout (float)

    Returns:
        model (tf.keras.Model)
    """
    backbone_name = backbone_name.lower()
    if 'resnet' in backbone_name:
        base = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=input_shape)
    elif 'densenet' in backbone_name:
        base = tf.keras.applications.DenseNet121(weights='imagenet', include_top=False, input_shape=input_shape)
    elif 'vgg' in backbone_name:
        base = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=input_shape)
    else:
        raise ValueError('Unsupported backbone: ' + backbone_name)

    base.trainable = False

    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = models.Model(inputs=base.input, outputs=outputs)
    return model


def build_small_custom_cnn(input_shape=(224, 224, 3)):
    model = models.Sequential([
        layers.Conv2D(32, 3, activation='relu', input_shape=input_shape),
        layers.MaxPooling2D(2),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Conv2D(128, 3, activation='relu'),
        layers.MaxPooling2D(2),
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(1, activation='sigmoid')
    ])
    return model


if __name__ == '__main__':
    m = build_transfer_model('ResNet50')
    m.summary()
