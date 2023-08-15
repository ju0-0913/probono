import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

import matplotlib.pyplot as plt

def add_random_noise(x):
    x = x + np.random.normal(size=x.shape) * np.random.uniform(1, 5)
    x = x - x.min()
    x = x / x.max()

    return x * 255.0

TRAINING_DIR = "C:/Users/juyeo/OneDrive/바탕 화면/probono/train"
TEST_DIR = "C:/Users/juyeo/OneDrive/바탕 화면/probono/test"

BATCH_SIZE = 8

training_datagen = ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 10,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    brightness_range = (0.5, 1.3),
    horizontal_flip = True,
    fill_mode = 'nearest',
    preprocessing_function = add_random_noise)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_generator = training_datagen.flow_from_directory(
    TRAINING_DIR,
    batch_size = BATCH_SIZE,
    target_size = (224, 224),
    class_mode = 'categorical',
    shuffle = True
)

test_generator = test_datagen.flow_from_directory(
    TEST_DIR,
    batch_size = BATCH_SIZE,
    target_size = (224, 224),
    class_mode = 'categorical'
)

img, label = next(training_generator)
plt.figure(figsize = (20, 20))

for i in range(BATCH_SIZE):
    plt.subplot(3, 3, i+1)
    plt.imshow(img[i])
    plt.title(label[i])
    plt.axis('off')

plt.show()

base_model = tf.keras.applications.VGG16(input_shape = (224, 224, 3),
                                        include_top = False,
                                        weights = 'imagenet')

base_model.trainable = False

out_layer = tf.keras.layers.Conv2D(128, (1, 1), padding = 'SAME', activation = None)(base_model.output)
out_layer = tf.keras.layers.BatchNormalization()(out_layer)
out_layer = tf.keras.layers.ReLU()(out_layer)

out_layer = tf.keras.layers.GlobalAveragePooling2D()(out_layer)

out_layer = tf.keras.layers.Dense(3, activation = 'softmax')(out_layer)

model = tf.keras.models.Model(base_model.input, out_layer)

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(learning_rate = 0.001),
              metrics = ['accuracy'])

history = model.fit(training_generator, epochs = 25, validation_data = test_generator, verbose = 1)

model.save("saved_model.h5")