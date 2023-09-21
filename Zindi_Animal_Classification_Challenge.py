from google.colab import drive
drive.mount('/content/drive', force_remount=True)

import os
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from sklearn.model_selection import train_test_split

# Define the image dimensions
img_width = 224
img_height = 224
batch_size = 32
epochs = 10

# Define data directories
data_dir = '/content/drive/MyDrive/Data'
train_elephants_dir = os.path.join(data_dir, 'train_elephants')
train_zebras_dir = os.path.join(data_dir, 'train_zebras')
test_dir = os.path.join(data_dir, 'test')

# Load the images - Load a smaller sample for testing
def load_sample_images_from_directory(directory, sample_size=1000):
    image_list = []
    for i, filename in enumerate(os.listdir(directory)):
        if i >= sample_size:
            break  # Stop loading after the specified sample size
        if filename.endswith(('.jpg', '.jpeg', '.png', '.gif')):
            try:
                img = tf.keras.preprocessing.image.load_img(
                    os.path.join(directory, filename),
                    target_size=(img_width, img_height),
                    color_mode="rgb"
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                img_array /= 255.0
                image_list.append(img_array)
            except Exception as e:
                print(f"Skipping invalid image: {os.path.join(directory, filename)}")
                print(f"Error: {e}")
                continue
    return np.array(image_list)

# Load a smaller sample of labels and combine data
sample_size = 1000  # Adjust the sample size as needed
train_elephants_imgs = load_sample_images_from_directory(train_elephants_dir, sample_size)
train_zebras_imgs = load_sample_images_from_directory(train_zebras_dir, sample_size)
test_imgs = load_sample_images_from_directory(test_dir, sample_size)

# Create labels for the sample
train_elephants_labels = np.zeros(len(train_elephants_imgs), dtype=int)
train_zebras_labels = np.ones(len(train_zebras_imgs), dtype=int)

X_sample = np.concatenate((train_elephants_imgs, train_zebras_imgs), axis=0)
y_sample = np.concatenate((train_elephants_labels, train_zebras_labels), axis=0)

# Split the sample data into training and validation sets
X_train_sample, X_val_sample, y_train_sample, y_val_sample = train_test_split(
    X_sample, y_sample, test_size=0.2, random_state=42)

# Initialize cross-validation
n_splits = 5  # You can adjust this number
skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
predictions_ensemble = []

# Perform cross-validation
for fold, (train_index, val_index) in enumerate(skf.split(X_train_sample, y_train_sample)):
    print(f"Training fold {fold + 1}...")

    X_train, X_val = X_train_sample[train_index], X_train_sample[val_index]
    y_train, y_val = y_train_sample[train_index], y_train_sample[val_index]

    # Data augmentation for training
    train_datagen = ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        shear_range=0.2,
        zoom_range=0.2,
        fill_mode='nearest'
    )

    # Define MobileNetV2 base model
    base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    base_model.trainable = True

    # Create a custom model on top of the base model
    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Data generators
    train_generator = train_datagen.flow(X_train, y_train, batch_size=batch_size)
    val_datagen = ImageDataGenerator()
    val_generator = val_datagen.flow(X_val, y_val, batch_size=batch_size)

    # Train the model
    model.fit(
        train_generator,
        steps_per_epoch=len(X_train) // batch_size,
        validation_data=val_generator,
        validation_steps=len(X_val) // batch_size,
        epochs=epochs
    )

    # Predict on the test data
    test_datagen = ImageDataGenerator()
    test_generator = test_datagen.flow(test_imgs, batch_size=1, shuffle=False)
    predictions = model.predict(test_generator)


# Define data directories
data_dir = '/content/drive/MyDrive/Data'
train_elephants_dir = os.path.join(data_dir, 'train_elephants')
train_zebras_dir = os.path.join(data_dir, 'train_zebras')
test_dir = os.path.join(data_dir, 'test')

# Get a list of test filenames
test_filenames = [os.path.basename(file) for file in os.listdir(test_dir)]


# Initialize a list to store the results
results = []

# Load your trained model
model = load_model('/content/drive/MyDrive/Data/zebra_vs_elephant_model.h5')

# Iterate through all expected test filenames
for filename in expected_test_filenames:
    # Check if the filename is in the list of test filenames
    if filename in test_filenames:
        # The file is present, predict its label
        img = load_img(os.path.join(test_dir, filename), target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = img_array / 255.0  # Normalize the image
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension

        # Predict using the model
        prediction = model.predict(img_array)[0][0]
        label = 1 if prediction >= 0.5 else 0
    else:
        # The file is missing, assume label 0
        label = 0

    results.append([filename, label])

# Create a DataFrame from the results
submission = pd.DataFrame(results, columns=['id', 'label'])

# Save the submission to a CSV file
submission.to_csv('/content/drive/MyDrive/Data/zebra_vs_elephant_submission.csv', index=False)

