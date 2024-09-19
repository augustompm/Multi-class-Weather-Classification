import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import glob
from PIL import Image
import matplotlib.pyplot as plt
from collections import Counter
import seaborn as sns

curr_path = os.getcwd()
base_dir = os.path.join(curr_path, 'Multi-class Weather Dataset')
folders = os.listdir(base_dir)
print(folders)

img_size = 250
batch_size = 32
num_epochs = 15
num_classes = 4

# Labels files with Image Data Generator
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=0.2
)

# Set validation split
train_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='training'  # Set as training data
)

validation_generator = train_datagen.flow_from_directory(
    base_dir,  # Same directory as training data
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',  # Set as validation data
    shuffle=False  # Evita esgotar o gerador de validação
)

# Generated Labels
labels = train_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
print(labels)

# Function for plotting images
def plotimages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

# Setting seed and clearing session
tf.keras.backend.clear_session()
tf.random.set_seed(51)
np.random.seed(51)

# Defining model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, (5, 5), input_shape=(img_size, img_size, 3), activation='relu'),
    tf.keras.layers.MaxPool2D(3, 3),
    tf.keras.layers.Conv2D(64, (5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D((3, 3)),
    tf.keras.layers.Conv2D(128, (5, 5), activation='relu'),
    tf.keras.layers.MaxPool2D(3, 3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.5, seed=5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

# Getting model summary after compiling
model.compile(
    loss='categorical_crossentropy',
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)
model.summary()

# Defining callback for early stopping
class My_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') is not None and logs.get('accuracy') > 0.95 and
            logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.95):
            print('\nStopping training')
            self.model.stop_training = True

callbacks = My_callback()

# Training model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=num_epochs,
    callbacks=[callbacks]
)

# Saves model
model.save('weather_cnn.h5')

# Saves history into npy file
np.save('history_weather.npy', history.history)

# Displaying graphs
def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.title(string)
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()

plot_graphs(history, "accuracy")
plot_graphs(history, "loss")

def prediction(test_path):
    img = image.load_img(test_path, target_size=(img_size, img_size))
    img = image.img_to_array(img, dtype=np.uint8)
    img = np.array(img) / 255.0
    plt.title('Image')
    plt.axis('off')
    plt.imshow(img.squeeze())
    predict = model.predict(img[np.newaxis, ...])
    predicted_class = labels[np.argmax(predict[0], axis=-1)]
    print('Prediction Value: ', np.max(predict[0], axis=-1))
    print("Classified:", predicted_class)

test_path = os.path.join(curr_path, 'cloudy34.jpg')
prediction(test_path)
