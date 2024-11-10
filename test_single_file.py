import numpy as np
import tflite_runtime.interpreter as tflite
import common as com
from common import yaml_load, file_to_vector_array  # Import from common.py

def run_inference_on_audio(model_path, audio_file_path, config_path="baseline.yaml"):
    # Load YAML configuration
    param = com.yaml_load()

    # Extract features from the audio file
    feature_vector = file_to_vector_array(
        file_name=audio_file_path,
        n_mels=param['feature']['n_mels'],
        frames=param['feature']['frames'],
        n_fft=param['feature']['n_fft'],
        hop_length=param['feature']['hop_length'],
        power=param['feature']['power']
    )

    if feature_vector is None:
        print("Feature extraction failed.")
        return None

    # Initialize TFLite interpreter
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Set the tensor for input data
    interpreter.set_tensor(input_details[0]['index'], feature_vector.astype(np.float32))

    # Run inference
    interpreter.invoke()

    # Retrieve output data
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data

# Example usage
model_path = "pump_model.tflite"
audio_file_path = "anomaly_id_00_00000000.wav"
output = run_inference_on_audio(model_path, audio_file_path)
print("Output:", output)
