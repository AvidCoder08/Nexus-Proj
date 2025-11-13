"""
Minimal data utilities for the fracture detection project.
This file contains helper functions to build tf.data datasets from a directory
organized as:

dataset_root/
  train/
    fractured/
    normal/
  val/
    fractured/
    normal/
  test/
    fractured/
    normal/

It uses TensorFlow's image_dataset_from_directory for simplicity.
"""

from pathlib import Path
import tensorflow as tf


def build_image_datasets(data_dir, image_size=(224, 224), batch_size=32):
    """Create train/val/test datasets from a directory.

    Args:
        data_dir (str or Path): path containing train/val/test subfolders
        image_size (tuple): (height, width)
        batch_size (int)

    Returns:
        train_ds, val_ds, test_ds (tf.data.Dataset)
    """
    data_dir = Path(data_dir)
    train_dir = data_dir / 'train'
    val_dir = data_dir / 'val'
    test_dir = data_dir / 'test'

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        labels='inferred',
        label_mode='binary',
        image_size=image_size,
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=True,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        val_dir,
        labels='inferred',
        label_mode='binary',
        image_size=image_size,
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=False,
    )

    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        labels='inferred',
        label_mode='binary',
        image_size=image_size,
        batch_size=batch_size,
        color_mode='rgb',
        shuffle=False,
    )

    # Prefetch for performance
    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    return train_ds, val_ds, test_ds


if __name__ == '__main__':
    # Quick smoke test (won't run unless executed directly)
    print('Data utilities module')
