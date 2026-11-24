
import json
import numpy as np
import tensorflow as tf

from pathlib import Path
from PIL import Image


# Locate the project directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Load model configuration
CONFIG_PATH = PROJECT_ROOT / "models" / "metadata.json"


def load_metadata():
    with open(CONFIG_PATH, "r") as file:
        return json.load(file)


def load_prediction_model(model_name):
    """
    Load a trained GeoVision ML model.
    """

    model_path = (
        PROJECT_ROOT / "models" / f"geovision_{model_name}.keras"
    )

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = tf.keras.models.load_model(model_path)

    return model


def preprocess_uploaded_image(image, model_name):
    """
    Prepare an uploaded image for the selected model.
    """

    image = image.convert("RGB")

    if model_name == "cnn":
        image_size = (64, 64)

    elif model_name == "mobilenet":
        image_size = (96, 96)

    else:
        raise ValueError("Unknown model name")

    image = image.resize(image_size)

    image_array = np.array(image, dtype=np.float32)

    image_array = image_array / 255.0

    if model_name == "mobilenet":
        image_array = image_array * 2.0 - 1.0

    image_array = np.expand_dims(image_array, axis=0)

    return image_array


def predict_image(image, model, model_name, class_names):
    """
    Predict the land-cover category of one satellite image.
    """

    processed_image = preprocess_uploaded_image(
        image,
        model_name
    )

    probabilities = model.predict(
        processed_image,
        verbose=0
    )[0]

    predicted_index = int(np.argmax(probabilities))

    predicted_category = class_names[predicted_index]

    confidence = float(probabilities[predicted_index])

    return {
        "predicted_class": predicted_category,
        "confidence": confidence,
        "probabilities": {
            class_names[i]: float(probabilities[i])
            for i in range(len(class_names))
        }
    }
