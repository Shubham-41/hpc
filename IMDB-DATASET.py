

import tensorflow as td

from tensorflow import keras

import numpy as np


data = keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = data.load_data(num_words = 10000)

print(train_data[0])

# Mapping

word_index = data.get_word_index()

word_index = {k:(v+3) for k , v in word_index.items()}

word_index["<PAD>"] = 0

word_index["<START>"] = 1

word_index["<UNK>"] = 2 # unk stands for unknown

word_index["<UNUSED>"] = 3 

reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])


def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])

print(decode_review(test_data[0]))

print(len(test_data[0]), len(test_data[1]))

# this is not good for our work

# redefining our training and testing data

#preprocessing

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",maxlen=250)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",maxlen=250)


print(len(train_data[0]), len(test_data[1]))

# Defining the model

model = keras.Sequential()

model.add(keras.layers.Embedding(10000,16))

model.add(keras.layers.GlobalAveragePooling1D()) 

model.add(keras.layers.Dense(16,activation="relu"))

model.add(keras.layers.Dense(1,activation="sigmoid"))

model.summary()

model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]

x_train = train_data[10000:]

y_val = train_labels[:10000]

y_train = train_labels[10000:]


fitmodel= model.fit(x_train,y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val),verbose=1)


results = model.evaluate(test_data,test_labels)

print(results)

model.save("model.h5")

model = keras.models.load_model("model.h5")





