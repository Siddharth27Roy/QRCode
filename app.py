import tensorflow as tf
import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2

model = tf.keras.models.load_model("model/finalmodel.keras")

model.compile(
    optimizer='adam',
    loss=tf.keras.losses.BinaryCrossentropy(),  # Assuming binary classification
    metrics=[
        tf.keras.metrics.BinaryAccuracy(),  # Binary accuracy
        tf.keras.metrics.Precision(),        # Precision
        tf.keras.metrics.Recall(),           # Recall
        tf.keras.metrics.AUC()               # Area under the ROC curve
    ]
)

def saveUploadedImage(uploadedImage):
    try:
        with open(os.path.join('uploads',uploadedImage.name),'wb') as f:
            f.write(uploadedImage.getbuffer())
        return True
    except: 
        return False

def runModel(imgPath, model):
    image = cv2.imread(imgPath)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  
    image = cv2.resize(image, (224, 224))
    image = image / 255.0 

    image = np.expand_dims(image, axis=0)

    predictions = model.predict(image)

    predicted_class = 1 if predictions[0] > 0.5 else 0 

    class_probabilities = predictions[0]
    
    # return class_probabilities
    
    if predicted_class == 1:
        return "Benign"
    else:
        return "Malicious"

def deleteUploadedImage(uploadedImage):
    try:
        # Remove the uploaded file after processing
        os.remove(os.path.join('uploads', uploadedImage.name))
        return True
    except Exception as e:
        print("An error occurred:", e)
        return False

st.title('Hsfafaf')

uploadedImage = st.file_uploader('Choose an image')

if uploadedImage is not None:
    # save the image in uploads
    if saveUploadedImage(uploadedImage):
        # display image 
        displayImage = Image.open(uploadedImage)
        # process image 
        result = runModel(os.path.join('uploads',uploadedImage.name), model)
        # delete Image
        deleteUploadedImage(uploadedImage)
        
        # display
        st.header('Your Uploaded Image is:')
        st.image(displayImage)
        
        st.header('The image is:')
        st.header(result)
        