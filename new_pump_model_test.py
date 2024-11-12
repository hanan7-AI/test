import numpy as np
import tensorflow as tf
import os
import common as com  # Assuming common functions are in this module

def test_single_file_with_tflite(audio_file_path, tflite_file_path):
    # Load parameters from the configuration file (baseline.yaml)
    param = com.yaml_load()

    # Extract features from the audio file using the same parameters as the training process
    data = com.file_to_vector_array(
        file_name=audio_file_path,
        n_mels=param["feature"]["n_mels"],
        frames=param["feature"]["frames"],
        n_fft=param["feature"]["n_fft"],
        hop_length=param["feature"]["hop_length"],
        power=param["feature"]["power"]
    )

    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_file_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Prepare the data (ensure it matches the input shape of the model)
    input_data = np.array(data, dtype=np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the reconstructed output from the TFLite model
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Calculate the anomaly score by finding the reconstruction error
    errors = np.mean(np.square(data - output_data), axis=1)
    anomaly_score = np.mean(errors)

    # Output the anomaly score
    com.logger.info(f"Anomaly score for file {os.path.basename(audio_file_path)}: {anomaly_score}")
    return anomaly_score

if _name_ == "_main_":
    # File paths for TFLite model and audio file
    audio_file_path = r"C:\Users\alqahth\Downloads\dcase2020_task2_baseline\dcase2020_task2_baseline\dev_data\pump\test\anomaly_id_00_00000000.wav"
    
    # Get the anomaly score using the TFLite model
    anomaly_score = test_single_file_with_tflite(audio_file_path, tflite_file_path)
    print(f"Anomaly Score: {anomaly_score}")
