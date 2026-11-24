
import json
from pathlib import Path

import tensorflow as tf

from src.prediction import load_prediction_model, load_metadata


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "new_training_data"
MODEL_NAME = "mobilenet"


def retrain_model(epochs=5, batch_size=32):

    metadata = load_metadata()
    class_names = metadata["image_classes"]

    # Check that all class folders exist
    missing_classes = [
        name for name in class_names
        if not (DATA_DIR / name).is_dir()
    ]

    if missing_classes:
        raise ValueError(
            f"Missing class folders: {missing_classes}"
        )

    # Load new images
    dataset = tf.keras.utils.image_dataset_from_directory(
        DATA_DIR,
        labels="inferred",
        label_mode="int",
        class_names=class_names,
        image_size=(96, 96),
        batch_size=batch_size,
        shuffle=True,
        seed=42
    )

    # Apply MobileNetV2 preprocessing
    def preprocess(images, labels):
        images = tf.cast(images, tf.float32) / 255.0
        images = images * 2.0 - 1.0
        return images, labels

    dataset = dataset.map(
        preprocess,
        num_parallel_calls=tf.data.AUTOTUNE
    ).prefetch(tf.data.AUTOTUNE)

    # Load existing model
    model = load_prediction_model(MODEL_NAME)

    # Continue training from existing weights
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        dataset,
        epochs=epochs
    )

    # Save updated model
    model_path = (
        PROJECT_ROOT / "models" /
        f"geovision_{MODEL_NAME}.keras"
    )

    model.save(model_path)

    return {
        "message": "Model retraining completed",
        "model": MODEL_NAME,
        "epochs": epochs,
        "final_accuracy": float(
            history.history["accuracy"][-1]
        ),
        "saved_model": str(model_path)
    }
