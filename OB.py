#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

# Function to generate synthetic images with random rectangles
def generate_synthetic_data(num_samples, img_size=(128, 128)):
    images = []
    labels = []
    
    for _ in range(num_samples):
        # Create a blank image
        image = np.ones((img_size[0], img_size[1], 3))
        
        # Randomly generate a rectangle (bounding box)
        x_min, y_min = np.random.randint(0, img_size[0] // 2, size=2)
        x_max, y_max = np.random.randint(img_size[0] // 2, img_size[0], size=2)
        
        # Add the rectangle to the image (just for visualization)
        image[y_min:y_max, x_min:x_max] = np.random.random(3)  # Random color
        
        # Normalize bounding box coordinates to [0, 1]
        bbox = [x_min / img_size[0], y_min / img_size[1], x_max / img_size[0], y_max / img_size[1]]
        
        images.append(image)
        labels.append(bbox)
    
    images = np.array(images)
    labels = np.array(labels)
    
    return images, labels

# Generate synthetic data for training
train_images, train_labels = generate_synthetic_data(100)

# Load the pre-trained MobileNetV2 model
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(128, 128, 3))

# Add custom layers for object detection
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(4, activation='sigmoid')(x)  # Bounding box coordinates

# Create the full model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the base model layers
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(train_images, train_labels, epochs=10, batch_size=16, validation_split=0.2)

# Save the model
model.save('object_detection_model.h5')




# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('object_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)


    
    
    
import cv2
import tensorflow.lite as tflite

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path='object_detection_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Start the video stream
cap = cv2.VideoCapture(0)  # Replace '0' with the correct index for your USB camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the model's input size (128x128)
    img_resized = cv2.resize(frame, (128, 128))

    # Preprocess the image for the model
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the predicted bounding box
    output_data = interpreter.get_tensor(output_details[0]['index'])
    bbox = output_data[0]  # Bounding box coordinates: [x_min, y_min, x_max, y_max]

    # Scale the bounding box coordinates back to the original image size
    h, w, _ = frame.shape
    x_min = int(bbox[0] * w)
    y_min = int(bbox[1] * h)
    x_max = int(bbox[2] * w)
    y_max = int(bbox[3] * h)

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# In[2]:


# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('object_detection_model.tflite', 'wb') as f:
    f.write(tflite_model)


# In[4]:


import cv2
import tensorflow.lite as tflite

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path='object_detection_model.tflite')
interpreter.allocate_tensors()

# Get input and output details
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Start the video stream
cap = cv2.VideoCapture(0)  # Replace '0' with the correct index for your USB camera

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        break

    # Resize the frame to the model's input size (128x128)
    img_resized = cv2.resize(frame, (128, 128))

    # Preprocess the image for the model
    input_data = np.expand_dims(img_resized, axis=0).astype(np.float32)

    # Set the input tensor
    interpreter.set_tensor(input_details[0]['index'], input_data)

    # Run inference
    interpreter.invoke()

    # Get the predicted bounding box
    output_data = interpreter.get_tensor(output_details[0]['index'])
    bbox = output_data[0]  # Bounding box coordinates: [x_min, y_min, x_max, y_max]

    # Scale the bounding box coordinates back to the original image size
    h, w, _ = frame.shape
    x_min = int(bbox[0] * w)
    y_min = int(bbox[1] * h)
    x_max = int(bbox[2] * w)
    y_max = int(bbox[3] * h)

    # Draw the bounding box on the frame
    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Object Detection', frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break



# In[ ]:




