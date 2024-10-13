#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import tflite_runtime.interpreter as tflite
import numpy as np


# In[ ]:


# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path="one_layer_model.tflite")

# Allocate tensors
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Create random input data (assuming input is float32)
input_data = np.random.random_sample(input_details[0]['shape']).astype(np.float32)

# Set input tensor
interpreter.set_tensor(input_details[0]['index'], input_data)

# Run inference
interpreter.invoke()

# Get the output tensor
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the output
print(output_data)

