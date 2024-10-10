#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tensorflow as tf
import numpy as np

# Load and run the TFLite model
interpreter = tf.lite.Interpreter(model_path="one_layer_model.tflite")
interpreter.allocate_tensors()

# Create a random input tensor based on the model's input shape
input_details = interpreter.get_input_details()
input_data = np.random.random_sample(input_details[0]['shape']).astype(np.float32)

# Run inference
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get and print the output
output_details = interpreter.get_output_details()
output_data = interpreter.get_tensor(output_details[0]['index'])
print(output_data)


# In[ ]:




