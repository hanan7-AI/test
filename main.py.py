#!/usr/bin/env python
# coding: utf-8

# In[3]:


# main.py
import model
import convert_model
import inference

def main():
    model_path = 'object_detection_model.h5'
    tflite_path = 'object_detection_model.tflite'
    
    # Create and train the model
    trained_model = model.create_model()
    trained_model.compile(optimizer='adam', loss='mean_squared_error')
    train_images, train_labels = generate_synthetic_data(100)
    trained_model.fit(train_images, train_labels, epochs=10, batch_size=16, validation_split=0.2)
    trained_model.save(model_path)
    
    # Convert the model to TensorFlow Lite
    convert_model.convert_to_tflite(model_path, tflite_path)
    
    # Load the TFLite model and start the video stream
    interpreter = inference.load_tflite_model(tflite_path)
    inference.start_video_stream(interpreter)

if __name__ == "__main__":
    main()


# In[ ]:





# In[ ]:




