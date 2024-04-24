#!/usr/bin/env python
# coding: utf-8

# In[1]:


# IMDB DATASET

# 25000 reviews for training

# 25000 revies for testing


# In[3]:


import tensorflow as td

from tensorflow import keras

import numpy as np


# In[5]:


data = keras.datasets.imdb

(train_data,train_labels),(test_data,test_labels) = data.load_data(num_words = 10000)


# In[6]:


print(train_data[0])


# In[13]:


# Mapping

word_index = data.get_word_index()


# In[14]:


word_index = {k:(v+3) for k , v in word_index.items()}

word_index["<PAD>"] = 0

word_index["<START>"] = 1

word_index["<UNK>"] = 2 # unk stands for unknown

word_index["<UNUSED>"] = 3 


# In[16]:


reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])


# In[17]:


def decode_review(text):
    return " ".join([reverse_word_index.get(i,"?") for i in text])


# In[18]:


print(decode_review(test_data[0]))


# In[20]:


print(len(test_data[0]), len(test_data[1]))

# this is not good for our work


# In[21]:


# redefining our training and testing data


# In[26]:


#preprocessing

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post",maxlen=250)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post",maxlen=250)


# In[27]:


print(len(train_data[0]), len(test_data[1]))


# In[28]:


# Defining the model

model = keras.Sequential()

model.add(keras.layers.Embedding(10000,16))

model.add(keras.layers.GlobalAveragePooling1D()) 

model.add(keras.layers.Dense(16,activation="relu"))

model.add(keras.layers.Dense(1,activation="sigmoid"))


# In[29]:


model.summary()


# In[31]:


model.compile(optimizer="adam",loss="binary_crossentropy", metrics=["accuracy"])


# In[32]:


x_val = train_data[:10000]

x_train = train_data[10000:]


# In[33]:


y_val = train_labels[:10000]

y_train = train_labels[10000:]


# In[34]:


fitmodel= model.fit(x_train,y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val),verbose=1)


# In[37]:


results = model.evaluate(test_data,test_labels)


# In[38]:


print(results)


# In[39]:


model.save("model.h5")


# In[40]:


model = keras.models.load_model("model.h5")


# In[ ]:




