import tensorflow as tf
from keras import models, layers
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import numpy as np
from keras.callbacks import EarlyStopping
# from IPython.display import HTML

# Define constants
IMAGE_SIZE = 224  # Adjust according to your requirements
BATCH_SIZE = 32   # Adjust according to your requirements

# Define paths to your datasets
train_directory = r'data\Training'
test_directory = r'data\Testing'
validation_directory = r'data\Validation'

# Load training dataset from directory
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"  # Remove label_mode='binary'
)


# Load training dataset from directory
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    train_directory,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"  # Remove label_mode='binary'
)

# Load testing dataset from directory
test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    test_directory,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"  # Remove label_mode='binary'
)

# Load validation dataset from directory
validation_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    validation_directory,
    shuffle=True,
    image_size=(IMAGE_SIZE, IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    color_mode="grayscale"  # Remove label_mode='binary'
)

# Flatten labels
train_labels_flat = np.concatenate([labels.numpy() for _, labels in train_dataset], axis=0)
test_labels_flat = np.concatenate([labels.numpy() for _, labels in test_dataset], axis=0)
validation_labels_flat = np.concatenate([labels.numpy() for _, labels in validation_dataset], axis=0)

# Print shapes of flattened labels
print("Shape of flattened training labels:", train_labels_flat.shape)
print("Shape of flattened testing labels:", test_labels_flat.shape)
print("Shape of flattened validation labels:", validation_labels_flat.shape)



# Define the model architecture
model = models.Sequential([
    # Input layer (Rescaling layer not needed in code as it's applied during preprocessing)
    layers.Conv2D(16, kernel_size=(3, 3), padding='same', activation='relu', input_shape=(224, 224, 1)),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Conv2D(32, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(64, kernel_size=(3, 3), padding='same', activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),  
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')  # Assuming 2 output classes
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Print the model summary
model.summary()

# Define EarlyStopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',  # Monitor validation loss for early stopping
    patience=5,           # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True  # Restore the weights of the best epoch
)

# Assuming you have your training data (train_dataset) and validation data (validation_dataset) prepared already
# Train the model with early stopping
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=30,
    callbacks=[early_stopping]  # Pass the EarlyStopping callback
)

# After training, you can evaluate the model on the test dataset
test_loss, test_accuracy = model.evaluate(test_dataset)

print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

model.save('finalmodel.keras')

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

test_loss, binary_accuracy, precision, recall, auc = model.evaluate(test_dataset)