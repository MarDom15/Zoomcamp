import tensorflow as tf
import os

os.chdir("/home/student/Dokumente/TensorFlow/workspace/plantTraining6/exported-models/")

# Convert the model
converter = tf.lite.TFLiteConverter.from_saved_model("/home/student/Dokumente/TensorFlow/workspace/plantTraining6/exported-models/saved_model")

#SavedModel directory
tflite_model = converter.convert()

# Save the model.
with open('model.tflite', 'wb') as f:
  f.write(tflite_model)

