#!/usr/bin/env python
# coding: utf-8

# In[1]:


# inference.py
import cv2
import numpy as np
import tensorflow.lite as tflite

def load_tflite_model(model_path):
    interpreter = tflite.Interpreter(model_path=model_path)
    interpreter.allocate_tensors()
    return interpreter

def run_inference(interpreter, frame):
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Resize and preprocess the frame
    img_resized = cv2.resize(frame, (128, 128))
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)
    
    # Perform the prediction
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()
    output_data = interpreter.get_tensor(output_details[0]['index'])
    return output_data[0]  # Return bounding box

def start_video_stream(interpreter):
    cap = cv2.VideoCapture(0)  # USB webcam index

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        bbox = run_inference(interpreter, frame)
        x_min, y_min, x_max, y_max = [int(bbox[i] * frame.shape[i % 2]) for i in range(4)]
        
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow('Object Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


# In[ ]:




