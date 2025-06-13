import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models

# 1) Ruta a tu carpeta de imágenes (dataset)
data_dir = "../dataset"  # Ajusta la ruta si tu carpeta se llama diferente

# 2) Parámetros de imagen
img_height, img_width = 224, 224
batch_size = 32

# 3) Preparamos los datos (20% para validación)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_ds = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'
)

val_ds = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation'
)

# 4) Definimos un modelo CNN sencillo
model = models.Sequential([
    layers.Input(shape=(img_height, img_width, 3)),
    layers.Conv2D(32, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(train_ds.num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 5) Entrenamos el modelo
model.fit(train_ds, validation_data=val_ds, epochs=10)

# 6) Guardamos el modelo en formato Keras (.h5)
model.save("../assets/lector_taza_modelo.h5")
