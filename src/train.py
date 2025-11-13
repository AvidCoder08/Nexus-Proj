"""
Minimal training harness. This script is intentionally minimal and intended to be
customized by the team. It uses the data and model utilities.

Usage:
    python src/train.py --data_dir /path/to/dataset --epochs 10

"""

import argparse
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

from src.data import build_image_datasets
from src.model import build_transfer_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--data_dir', type=str, required=True)
    p.add_argument('--backbone', type=str, default='ResNet50')
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--output', type=str, default='models')
    return p.parse_args()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds, val_ds, test_ds = build_image_datasets(args.data_dir, batch_size=args.batch_size)

    model = build_transfer_model(backbone_name=args.backbone)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    ckpt = ModelCheckpoint(str(output_dir / 'best_model.h5'), save_best_only=True, monitor='val_loss')
    es = EarlyStopping(patience=5, restore_best_weights=True)
    rlrop = ReduceLROnPlateau(patience=3)

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        callbacks=[ckpt, es, rlrop]
    )

    model.save(str(output_dir / 'final_model.h5'))


if __name__ == '__main__':
    main()
