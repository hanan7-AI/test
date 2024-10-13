
import numpy as np
import soundfile as sf
import tflite_runtime.interpreter as tflite
from scipy.signal import resample


def preprocess_audio(file_path):
    audio_data, sample_rate = sf.read(file_path)  
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
input_audio = preprocess_audio("anomaly_id_00_00000008.wav")

# Reshape input to match the model's expected 
input_audio = input_audio[:10] 
input_audio = input_audio.reshape(1, 10)

# input tensor
interpreter.set_tensor(input_details[0]['index'], input_audio)

# inference
interpreter.invoke()

# Get the output result
output_data = interpreter.get_tensor(output_details[0]['index'])

# Print the result
print("Model output:", output_data)

# no fix value!
if output_data[0] > 0.5:
    print("Anomaly detected")
else:
    print("Normal audio")

