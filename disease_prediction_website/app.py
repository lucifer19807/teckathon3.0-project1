from flask import Flask, render_template, request, redirect, url_for
import numpy as np
from PIL import Image
import io
import cv2
import tensorflow as tf
from pathlib import Path

app = Flask(__name__)

new_model = tf.keras.models.load_model('../Tuberculosis_model/my_model1')

def predict_tuberculosis(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (28, 28))
    if img.shape[2] == 1:
        img = np.dstack([img, img, img])
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.array(img)
    img = img / 255
    test_data = []
    test_data.append(img)
    test_data = np.array(test_data)
    predictions = new_model.predict(test_data)
    
    
    if np.argmax(predictions):
        return "Yes ,Tuberculosis is there "
    elif np.argmax(predictions)!= 1:
        return "No ,Tuberculosis is not there "
        
    

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            img = Image.open(io.BytesIO(file.read()))
            temp_image_path = 'temp_image.jpg'
            img.save(temp_image_path)
            prediction_result = predict_tuberculosis(temp_image_path)
            return render_template('prediction.html', result=prediction_result)
    return render_template('upload.html')

if __name__ == '__main__':
    app.run(debug=True)
