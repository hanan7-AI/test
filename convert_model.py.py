#!/usr/bin/env python
# coding: utf-8

# In[2]:


# convert_model.py
import tensorflow as tf

def convert_to_tflite(model_path, output_path):
    model = tf.keras.models.load_model(model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(output_path, 'wb') as f:
        f.write(tflite_model)
    print(f"Model converted to TensorFlow Lite and saved at {output_path}")


# In[4]:


print(output_path)


# In[ ]:




