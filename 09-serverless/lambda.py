import numpy as np
from PIL import Image
from io import BytesIO
from urllib import request
import tflite_runtime.interpreter as tflite

# Télécharger et préparer l'image
def download_image(url):
    with request.urlopen(url) as resp:
        buffer = resp.read()
    stream = BytesIO(buffer)
    img = Image.open(stream)
    return img

def prepare_image(img, target_size):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    img = img.resize(target_size, Image.NEAREST)
    return np.expand_dims(np.array(img) / 255.0, axis=0)

def lambda_handler(event, context):
    image_url = event["url"]
    img = download_image(image_url)
    img_array = prepare_image(img, (200, 200))

    # Charger le modèle
    interpreter = tflite.Interpreter(model_path=r"C:\Users\marti\Desktop\Nouveau dossier\Homework_9\model_2024_hairstyle_v2.tflite")
    interpreter.allocate_tensors()

    # Exécuter le modèle
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    return {"prediction": float(output_data[0][0])}
