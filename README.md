# Cats vs. Dogs Image Classification using a Custom VGG-like Model

## Table of Contents
1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Model Architecture](#model-architecture)
4. [Dataset](#dataset)
5. [Setup](#setup)
6. [Usage](#usage)
7. [Results](#results)
8. [Contributing](#contributing)
9. [License](#license)

## 1. Project Overview
This repository presents an image classification solution for distinguishing between cat and dog images. The project leverages TensorFlow and Keras to implement a custom convolutional neural network (CNN) inspired by the VGG architecture. The model is trained on the `cats_vs_dogs` dataset, demonstrating a complete pipeline from data loading and preprocessing to model training and evaluation.

## 2. Features
- **Efficient Data Loading**: Utilizes `tf.data` for optimized data loading, resizing, and normalization.
- **Custom VGG-like Architecture**: Implements a deep CNN with `Conv2D`, `BatchNormalization`, `ReLU`, and `MaxPooling2D` layers.
- **Keras API Integration**: Seamlessly integrates with Keras for model definition, compilation, and training.
- **Early Stopping**: Incorporates `EarlyStopping` callback to prevent overfitting and optimize training duration.
- **Model Persistence**: Saves the trained model for future inference.

## 3. Model Architecture
The core of this project is a custom VGG-like convolutional neural network. The architecture consists of five convolutional blocks, each followed by a `MaxPooling2D` layer. Each convolutional block contains multiple `Conv2D` layers, `BatchNormalization` for stable training, and `ReLU` activation functions. The number of filters increases with depth, starting from 64 and going up to 512.

Following the convolutional blocks, the output is flattened and passed through two fully connected (Dense) layers with 4096 units each, using `ReLU` activation. The final output layer is a `Dense` layer with 1 unit and a `sigmoid` activation function, suitable for binary classification.

**Block Structure:**
- **Input Layer**: `(224, 224, 3)`
- **Block 1**: 2 x `Conv2D(64)`, `BatchNormalization`, `ReLU`, `MaxPooling2D`
- **Block 2**: 2 x `Conv2D(128)`, `BatchNormalization`, `ReLU`, `MaxPooling2D`
- **Block 3**: 3 x `Conv2D(256)`, `BatchNormalization`, `ReLU`, `MaxPooling2D`
- **Block 4**: 3 x `Conv2D(512)`, `BatchNormalization`, `ReLU`, `MaxPooling2D`
- **Block 5**: 4 x `Conv2D(512)`, `BatchNormalization`, `ReLU`, `MaxPooling2D`
- **Flatten Layer**
- **Fully Connected Layers**: 2 x `Dense(4096, activation='relu')`
- **Output Layer**: `Dense(1, activation='sigmoid')`

## 4. Dataset
The project utilizes the `cats_vs_dogs` dataset from TensorFlow Datasets (`tfds`). This dataset contains images of cats and dogs, split into training and testing sets. For this project, the dataset is split as follows:
- **Training Set**: 80% of the original `train` split.
- **Test Set**: The remaining 20% of the original `train` split.

Images are preprocessed to a uniform size of `(224, 224)` and pixel values are normalized to the range `[0, 1]`.

## 5. Setup
To set up the environment and run the code, follow these steps:

### Prerequisites
- Python 3.8+
- pip (Python package installer)

### Installation
1. **Clone the repository (if applicable)**:
   ```bash
   git clone <repository_url>
   cd <repository_name>
   ```

2. **Install required packages**:
   ```bash
   pip install tensorflow tensorflow-datasets matplotlib keras
   ```

## 6. Usage
1. **Data Loading and Preprocessing**:
   The script automatically loads the `cats_vs_dogs` dataset, splits it into training and testing sets, resizes images to `224x224`, and normalizes pixel values.
   ```python
   import tensorflow as tf
   import tensorflow_datasets as tfds

   (train_dataset, test_dataset), info = tfds.load(
       'cats_vs_dogs',
       split=('train[:80%]', 'train[80%:]'),
       with_info=True,
       as_supervised=True
   )

   def normalize_img(image, label):
     return (tf.cast(image, tf.float32) / 255.0, label)

   def resize(image, label):
     return (tf.image.resize(image, (224, 224)), label)

   # Apply preprocessing and batching
   BATCH_SIZE = 4
   train_dataset = train_dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
   train_dataset = train_dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
   train_dataset = train_dataset.shuffle(len(train_dataset) // 1000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

   test_dataset = test_dataset.map(resize, num_parallel_calls=tf.data.AUTOTUNE)
   test_dataset = test_dataset.map(normalize_img, num_parallel_calls=tf.data.AUTOTUNE)
   test_dataset = test_dataset.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
   ```

2. **Model Definition and Compilation**:
   The custom VGG-like model is defined using Keras functional API. The model is compiled with `BinaryCrossentropy` loss and the `Adam` optimizer.
   ```python
   from tensorflow.keras.models import Model
   from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, Flatten, Dense, Input
   from tensorflow.keras.optimizers import Adam
   from tensorflow.keras.losses import BinaryCrossentropy

   # (vgg_block function and model definition as in the notebook)
   # ...
   model = Model(inputs = inp, outputs = output)

   model.compile(loss=BinaryCrossentropy(), optimizer=Adam(), metrics=['accuracy'])
   ```

3. **Training the Model**:
   The model is trained using the `fit` method with `EarlyStopping` callback to monitor validation loss.
   ```python
   from tensorflow.keras.callbacks import EarlyStopping

   earlystopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

   history = model.fit(
       train_dataset,
       epochs=100,
       validation_data=test_dataset,
       callbacks=[earlystopping]
   )
   ```

4. **Saving the Model**:
   The trained model weights and architecture are saved to an HDF5 file.
   ```python
   model.save('model.hgg')
   ```

