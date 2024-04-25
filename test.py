import tensorflow as tf
from keras import models, layers

model = tf.keras.models.load_model('april23th.h5')

test_directory = r'data\Testing'
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255.0)

test_generator = test_datagen.flow_from_directory(
    test_directory,
    target_size=(224, 224),
    batch_size=32,
    color_mode='grayscale',  # Use grayscale images
    class_mode='binary'
)

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

test_loss, binary_accuracy, precision, recall, auc = model.evaluate(test_generator)