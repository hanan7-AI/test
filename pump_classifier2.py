#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite
from scipy.signal import resample


# In[ ]:


def preprocess_audio(file_path):
    audio_data, sample_rate = sf.read(file_path)
    
    # Resample if the sample rate is different from the model's expected rate 
    if sample_rate != target_sample_rate:
        audio_data = resample(audio_data, int(len(audio_data) * target_sample_rate / sample_rate))
    # Ensure audio length is what the model expects
    if len(audio_data) < desired_length:
        # If too short, pad with zeros
        padding = desired_length - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), 'constant')
    else:
        # If too long, truncate
        audio_data = audio_data[:desired_length]
    
    # Normalize audio 
    audio_data = np.clip(audio_data, -1.0, 1.0)
    return np.array(audio_data, dtype=np.float32)

# Load the TFLite model
interpreter = tflite.Interpreter(model_path="pump_model.tflite")
interpreter.allocate_tensors()

# Get input and output tensor details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Load and preprocess the WAV file
input_audio = preprocess_audio("anomaly_id_00_00000009.wav")

# Reshape input to match the model's expected input shape (e.g., [1, 16000] if it's a single-channel audio input)
input_audio = input_audio.reshape(input_details[0]['shape'])

# Set the input tensor
interpreter.set_tensor(input_details[0]['index'], input_audio)

# Run inference
interpreter.invoke()

# Get the output result
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the result
print("Model output:", output_data)

# Assuming the output is a classification result (0 = normal, 1 = anomaly)
if output_data[0] > 0.5:
    print("Anomaly detected")
else:
    print("Normal audio")

