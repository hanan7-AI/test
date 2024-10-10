#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[3]:


# Load the existing HDF5 model
model = tf.keras.models.load_model(r'C:\Users\hanan\Desktop\testTF/model_pump.hdf5')

# Convert the model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TFLite model to a file
with open(r'C:\Users\hanan\Desktop\testTF/model_pump.tflite', 'wb') as f:
    f.write(tflite_model)


# In[ ]:




