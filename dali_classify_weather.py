import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
from nvidia.dali.pipeline import pipeline_def
import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.plugin.tf import DALIDataset
from collections import Counter

# Paths and constants
curr_path = os.getcwd()
base_dir = os.path.join(curr_path, 'Multi-class Weather Dataset')
folders = os.listdir(base_dir)
print(folders)

img_size = 250
batch_size = 16
num_epochs = 15
num_classes = 4

@pipeline_def
def dali_pipeline():
    jpegs, labels = fn.readers.file(file_root=base_dir, random_shuffle=True)
    
    # Decode the images to RGB format
    images = fn.decoders.image(jpegs, device="mixed", output_type=types.RGB)
    
    # Resize the images to a uniform size (128x128)
    images = fn.resize(images, resize_x=img_size, resize_y=img_size)
    
    # Apply normalization and ensure the output layout is HWC
    images = fn.crop_mirror_normalize(images, device="gpu",
                                      dtype=types.FLOAT, std=[255.0], mean=[0.485 * 255, 0.456 * 255, 0.406 * 255],
                                      output_layout="HWC")  # Specify the correct output layout
    
    # Move the labels to the GPU
    labels = labels.gpu()
    
    return images, labels


# Build the DALI pipeline
pipe = dali_pipeline(batch_size=batch_size, num_threads=4, device_id=0)
pipe.build()

def one_hot_encode_labels(labels, num_classes):
    return tf.one_hot(labels, num_classes)

# Convert DALI pipeline into a TensorFlow dataset and apply one-hot encoding
train_dataset = DALIDataset(
    pipeline=pipe,
    batch_size=batch_size,
    output_shapes=((batch_size, img_size, img_size, 3), (batch_size,)),
    output_dtypes=(tf.float32, tf.int32),
    device_id=0
).map(lambda images, labels: (images, one_hot_encode_labels(labels, num_classes)))
# Validation using ImageDataGenerator (since validation is less performance-critical)
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255, validation_split=0.2)

validation_generator = train_datagen.flow_from_directory(
    base_dir,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode='categorical',
    subset='validation',
    shuffle=False
)

# Generated Labels
labels = validation_generator.class_indices
labels = dict((v, k) for k, v in labels.items())
print(labels)

# Function to plot images
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

# Defining the model
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

# Compile the model
model.compile(
    loss='categorical_crossentropy',  # Use sparse categorical crossentropy for integer labels
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    metrics=['accuracy']
)

model.summary()

# Callback for early stopping
class My_callback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if (logs.get('accuracy') is not None and logs.get('accuracy') > 0.95 and
            logs.get('val_accuracy') is not None and logs.get('val_accuracy') > 0.95):
            print('\nStopping training')
            self.model.stop_training = True

callbacks = My_callback()

# Training the model using the DALI dataset
history = model.fit(
    train_dataset,
    validation_data=validation_generator,
    epochs=num_epochs,
    steps_per_epoch=16,
    callbacks=[callbacks]
)

# Save the model
model.save('weather_cnn_dali.h5')

# Save the history into a numpy file
np.save('history_weather_dali.npy', history.history)

# Plotting training graphs
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

# Prediction function with test image
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
