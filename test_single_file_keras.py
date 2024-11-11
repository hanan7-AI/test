import os
import numpy as np
import sys
import matplotlib.pyplot as plt
import common as com
import keras_model

def test_single_file(audio_file_path, model_file_path):
    if not os.path.exists(audio_file_path):
        com.logger.error(f"Audio file not found: {audio_file_path}")
        sys.exit(-1)


    if not os.path.exists(model_file_path):
        com.logger.error(f"Model file not found: {model_file_path}")
        sys.exit(-1)

    # Load the trained model
    model = keras_model.load_model(model_file_path)
    model.summary()

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


    # Predict the anomaly score by calculating the reconstruction error
    errors = np.mean(np.square(data - model.predict(data)), axis=1)
    anomaly_score = np.mean(errors)

    # Output the anomaly score
    com.logger.info(f"Anomaly score for file {os.path.basename(audio_file_path)}: {anomaly_score}")
    return anomaly_score


if __name__ == "__main__":
    # the audio file path and model path directly
    model_file_path = r"C:\ASD\dcase2020_task2_baseline\model\model_slider.hdf5"
    audio_file_path = r"C:\ASD\dcase2020_task2_baseline\dev_data\slider\test\anomaly_id_00_00000000.wav"

    # get the anomaly score
    anomaly_score = test_single_file(audio_file_path, model_file_path)
    print(f"Anomaly Score: {anomaly_score}")