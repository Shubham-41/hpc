
import tensorflow as tf

image_data = tf.keras.datasets.fashion_mnist


(train_images,train_labels),(test_images,test_labels) = image_data.load_data()


train_images[0]

# view of list of list 

import matplotlib.pyplot as plt

plt.imshow(train_images[0])

train_labels[0]


# # Preprocessing

train_images = train_images / 255.0

test_images = test_images / 255.0

train_images[0]

from tensorflow import keras

model = tf.keras.Sequential()

# image is of 28*28 pixel

model.add(keras.layers.Flatten(input_shape=(28,28)))

# neurons
model.add(keras.layers.Dense(128,activation="relu"))

# output of the network
model.add(keras.layers.Dense(10,activation='softmax'))

# how it is affecting the network to make the better predictions to optimize it

model.compile(optimizer="adam",loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=10,verbose=1)

model.predict(test_images)[0]

# list of 10 numbers 
# represent the probability of output

test_labels[0]

# Evaluate The model

test_loss, test_acc = model.evaluate(test_images,test_labels)

print('Test Accuracy:', test_acc)

# Make Predictions

import numpy as np

predictions = model.predict(test_images)

predicted_labels = np.argmax(predictions,axis=1)

# Show some example images and their predicted labels
num_rows = 5
num_cols = 5
num_images = num_rows * num_cols 
plt.figure(figsize=(2 * 2 *num_cols, 2 * num_rows))
for i in range(num_images):
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 1) 
    plt.imshow(test_images[i], cmap='gray')
    plt.axis('off')
    plt.subplot(num_rows, 2 * num_cols, 2 * i + 2) 
    plt.bar(range(10),predictions[i]) 
    plt.xticks(range(10))
    plt.ylim([0, 1])
    plt.tight_layout()
    plt.title(f"Predicted label: {predicted_labels[i]}")
    plt.show()

