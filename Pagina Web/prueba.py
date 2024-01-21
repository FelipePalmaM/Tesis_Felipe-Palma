import io
from flask import Flask, render_template, request, jsonify
import tensorflow as tf
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, preprocess_input
import numpy as np
from tensorflow.keras.preprocessing import image
# Ruta al modelo guardado
modelo_ruta = "C:/Users/palma/OneDrive/Escritorio/Pagina Web/Modelo_FInal_Resnet50"
modelo = tf.keras.models.load_model(modelo_ruta)
preprocess_input = tf.keras.applications.resnet_v2.preprocess_input

app = Flask(__name__)

# Esta es tu ruta existente
@app.route('/')
def index():
    return render_template('index.html')
# Esta es la nueva ruta para manejar la solicitud POST
@app.route('/procesar', methods=['POST'])
def procesar_imagen():
    image_path = request.files['imagen']
    imagen_bytesio = io.BytesIO(image_path.read())
    img = image.load_img(imagen_bytesio, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    predictions = modelo.predict(img_array).flatten()
    predictions = tf.nn.sigmoid(predictions)
    predictions = tf.where(predictions < 0.5, 0, 1)
    if(predictions==0):        
        respuesta = {'mensaje': 'Displasia'}
        return jsonify(respuesta)
    elif(predictions==1):        
        respuesta = {'mensaje': 'No Displasia'}
        return jsonify(respuesta)
    else:
        respuesta = {'mensaje': 'No funca'}
        return jsonify(respuesta)

if __name__ == '__main__':
    app.run(debug=True)
