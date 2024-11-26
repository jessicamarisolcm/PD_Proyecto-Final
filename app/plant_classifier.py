from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

# Cargar el modelo previamente entrenado
MODEL_PATH = "app/data/plant_model.h5"
model = load_model(MODEL_PATH)

def classify_plant(image_bytes):
    """Clasifica una planta a partir de una imagen."""
    image = Image.open(io.BytesIO(image_bytes)).resize((128, 128))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    classes = ["Rose", "Lily", "Cactus"]  # Asegúrate de usar tus clases reales
    return classes[np.argmax(prediction)]
