#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tensorflow import keras



# In[24]:


df = pd.read_csv("http://web.iitd.ac.in/~sumeet/A3/2016EE10459.csv",nrows = 3000)

data = np.array(df.values)
size = len(data) ;
le = int((len(data)*9)/10)

train_data = data[:le,:784]/255 ;
train_label = data[:le,784] ;

test_data = data[le:,:784]/255 ;
test_label = data[le:,784] ;


# In[29]:


model = keras.Sequential([
    keras.layers.Dense(100, activation='relu'),
    keras.layers.Dense(30, activation='relu'),
    keras.layers.Dense(10, activation='sigmoid')
])


# In[32]:


sgd  = keras.optimizers.SGD(lr=0.1)
model.compile(optimizer='sgd',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])


# In[34]:


model.fit(train_data, train_label,batch_size=1, epochs=20, verbose = 2)


# In[35]:


test_loss, test_acc = model.evaluate(test_data,  test_label, batch_size = 1, verbose = 2)

print('\nTest accuracy:', test_acc)


# In[ ]:




