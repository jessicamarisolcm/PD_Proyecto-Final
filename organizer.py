import os
import shutil

# Ruta base donde está tu dataset (ajustar según sea necesario)
base_dir = "PlantVillage"  # Cambia esto a tu ruta base

# Directorios de train y validation donde moveremos las imágenes
train_dir = os.path.join(base_dir, 'train')
val_dir = os.path.join(base_dir, 'validation')

# Lista de clases de plantas que tienes (esta es una lista general que puedes expandir si es necesario)
plant_classes = ['Apple', 'Bell Pepper', 'Cherry', 'Corn (Maize)', 'Grape', 'Peach', 'Potato', 'Strawberry', 'Tomato']

# Crear las carpetas principales de train y validation si no existen
if not os.path.exists(train_dir):
    os.makedirs(train_dir)
if not os.path.exists(val_dir):
    os.makedirs(val_dir)

# Función para mover imágenes de una planta específica
def move_images(plant_name):
    # Directorios de la planta
    plant_base_dir = os.path.join(base_dir, plant_name)
    train_plant_dir = os.path.join(plant_base_dir, 'Train')
    val_plant_dir = os.path.join(plant_base_dir, 'Val')

    # Crear las carpetas para las clases dentro de train y validation si no existen
    for folder in os.listdir(train_plant_dir):
        # Si la subcarpeta de la clase no existe en train, crearla
        class_train_dir = os.path.join(train_dir, folder)
        if not os.path.exists(class_train_dir):
            os.makedirs(class_train_dir)

        # Mover imágenes desde 'Train/clase' a 'train/clase'
        class_folder = os.path.join(train_plant_dir, folder)
        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)
            if os.path.isfile(file_path):
                shutil.move(file_path, os.path.join(class_train_dir, filename))

    for folder in os.listdir(val_plant_dir):
        # Si la subcarpeta de la clase no existe en validation, crearla
        class_val_dir = os.path.join(val_dir, folder)
        if not os.path.exists(class_val_dir):
            os.makedirs(class_val_dir)

        # Mover imágenes desde 'Val/clase' a 'validation/clase'
        class_folder = os.path.join(val_plant_dir, folder)
        for filename in os.listdir(class_folder):
            file_path = os.path.join(class_folder, filename)
            if os.path.isfile(file_path):
                shutil.move(file_path, os.path.join(class_val_dir, filename))

# Procesar todas las plantas
for plant in plant_classes:
    move_images(plant)
    print(f"Imágenes de {plant} reorganizadas correctamente.")

print("Reestructuración completada.")
