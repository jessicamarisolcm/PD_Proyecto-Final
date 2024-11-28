import tensorflow as keras
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras._tf_keras.keras.models import Sequential
from keras._tf_keras.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Rutas 
train_dir = "PlantVillage/train"
val_dir = "PlantVillage/validation"

# Generadores de imágenes
train_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_datagen = ImageDataGenerator(rescale=1.0/255.0)

train_generator = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode="categorical"
)
val_generator = val_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=32, class_mode="categorical"
)

# Modelo CNN
model = Sequential([
    Conv2D(32, (3, 3), activation="relu", input_shape=(128, 128, 3)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(128, activation="relu"),
    Dense(train_generator.num_classes, activation="softmax")
])

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Entrenamiento
model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Guardar el modelo
model.save("app/data/plant_model.h5")

print("Modelo entrenado y guardado como 'plant_model.h5'")
