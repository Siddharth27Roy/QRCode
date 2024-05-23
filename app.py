from flask import Flask, request, render_template, redirect
from werkzeug.utils import secure_filename
import tensorflow as tf
import numpy as np
import os
import cv2
from PIL import Image

# Initialize the Flask application
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'

# Load the model
model = tf.keras.models.load_model("model/finalmodel.keras")

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),  
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),  
        tf.keras.metrics.Precision(),       
        tf.keras.metrics.Recall(),          
        tf.keras.metrics.AUC()              
    ]
)

def save_uploaded_image(uploaded_image):
    try:
        filename = secure_filename(uploaded_image.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        uploaded_image.save(file_path)
        return file_path
    except Exception as e:
        print("An error occurred while saving the image:", e)
        return None

def run_model(img_path, model):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0

    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)

    predicted_class = 1 if predictions[0] > 0.5 else 0

    if predicted_class == 1:
        return "Benign"
    else:
        return "Malicious"

# def delete_uploaded_image(img_path):
#     try:
#         os.remove(img_path)
#         return True
#     except Exception as e:
#         print("An error occurred:", e)
#         return False

def delete_all_uploaded_images():
    folder = app.config['UPLOAD_FOLDER']
    for filename in os.listdir(folder):
        if filename != '.gitkeep':  # Skip deletion of .gitkeep file
            file_path = os.path.join(folder, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
            except Exception as e:
                print(f"An error occurred while deleting the file {file_path}: {e}")

@app.route('/')
def home():
    delete_all_uploaded_images()
    return render_template('home.html')

@app.route('/model', methods=['GET', 'POST'])
def upload_image():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            file_path = save_uploaded_image(file)
            if file_path:
                result = run_model(file_path, model)
                # delete_uploaded_image(file_path)
                return render_template('result.html', result=result, image_path=file_path)
    delete_all_uploaded_images()
    return render_template('upload.html')

if __name__ == "__main__":
    app.run(debug=True)
