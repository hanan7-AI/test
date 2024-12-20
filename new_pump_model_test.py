import numpy as np
import tflite_runtime.interpreter as tflite
import os
import common as com  

def get_anomaly_score(audio_file_path, tflite_file_path):
    # Load parameters from the configuration file (baseline.yaml)
    param = com.yaml_load()

    # Extract features from the audio file using the same parameters as the training process
    data = com.file_to_vector_array(
        file_name=audio_file_path,
        n_mels=param["feature"]["n_mels"],
        frames=param["feature"]["frames"] ,
        n_fft=param["feature"]["n_fft"],
        hop_length=param["feature"]["hop_length"],
        power=param["feature"]["power"]
    )

    print(data.shape)
    
    # Load the TFLite model and allocate tensors
    interpreter = tflite.Interpreter(model_path=tflite_file_path)
    interpreter.allocate_tensors()

    # Get input and output details
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    #print(input_details , output_details)
    # Prepare the data 
    # input_data = np.array(data, dtype=np.float32)
    #input_data = np.reshape(data,(1,10))
    #if input_data.shape[1] != 640 :
    #input_data= np.reshape(input_data, (1,640))
    

    
    interpreter.resize_tensor_input(input_details[0]['index'],[465,640])##########
    interpreter.allocate_tensors()
    
    interpreter.set_tensor(input_details[0]['index'], data.astype(np.float32))
    interpreter.invoke()

    # Get the output from the TFLite model
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # Calculate the anomaly score 
    errors = np.mean(np.square(data - output_data), axis=1)
    anomaly_score = np.mean(errors)

    return anomaly_score

# Get the anomaly score using the TFLite model

tflite_file_path = "model_pump_new.tflite"
audio_file_path = "normal_id_01_00000000.wav"

anomaly_score = get_anomaly_score(audio_file_path, tflite_file_path)
print(f"Anomaly Score: {anomaly_score}")
