import tensorflow as tf

# 1) Carga el modelo que acabas de entrenar
model = tf.keras.models.load_model("../assets/lector_taza_modelo.h5")

# 2) Configura el conversor a TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]  # lo hace más pequeño

# 3) Convierte
tflite_model = converter.convert()

# 4) Guarda el modelo TFLite en la carpeta assets
with open("../assets/lector_taza_modelo.tflite", "wb") as f:
    f.write(tflite_model)

print("¡Conversión completada! Verifica assets/lector_taza_modelo.tflite")
