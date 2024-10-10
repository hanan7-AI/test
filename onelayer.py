{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "3678e618-c28d-4d32-841d-9e9cd6f2b439",
   "metadata": {},
   "outputs": [],
   "source": [
    "import tensorflow as tf\n",
    "import os \n",
    "from tensorflow.keras.models import Sequential \n",
    "from tensorflow.keras.layers import Dense\n",
    "from tensorflow.keras.models import Model, load_model, save_model"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "c73b4c02-f818-493e-8628-e5f59025c406",
   "metadata": {},
   "outputs": [],
   "source": [
    "model = Sequential([Dense(units=1 , input_shape=(10,), activation='linear')])"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "cf75045f-93d5-418e-9d93-429a9f24b503",
   "metadata": {},
   "outputs": [],
   "source": [
    "model.compile(optimizer='adam' , loss='mean_squared_error')"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "a88b351a-b970-4352-a1ad-97c0639819bb",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\">Model: \"sequential_2\"</span>\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1mModel: \"sequential_2\"\u001b[0m\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\">┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃<span style=\"font-weight: bold\"> Layer (type)                         </span>┃<span style=\"font-weight: bold\"> Output Shape                </span>┃<span style=\"font-weight: bold\">         Param # </span>┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ dense_2 (<span style=\"color: #0087ff; text-decoration-color: #0087ff\">Dense</span>)                      │ (<span style=\"color: #00d7ff; text-decoration-color: #00d7ff\">None</span>, <span style=\"color: #00af00; text-decoration-color: #00af00\">1</span>)                   │              <span style=\"color: #00af00; text-decoration-color: #00af00\">11</span> │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n",
       "</pre>\n"
      ],
      "text/plain": [
       "┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓\n",
       "┃\u001b[1m \u001b[0m\u001b[1mLayer (type)                        \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1mOutput Shape               \u001b[0m\u001b[1m \u001b[0m┃\u001b[1m \u001b[0m\u001b[1m        Param #\u001b[0m\u001b[1m \u001b[0m┃\n",
       "┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩\n",
       "│ dense_2 (\u001b[38;5;33mDense\u001b[0m)                      │ (\u001b[38;5;45mNone\u001b[0m, \u001b[38;5;34m1\u001b[0m)                   │              \u001b[38;5;34m11\u001b[0m │\n",
       "└──────────────────────────────────────┴─────────────────────────────┴─────────────────┘\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Total params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">11</span> (44.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Total params: \u001b[0m\u001b[38;5;34m11\u001b[0m (44.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">11</span> (44.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Trainable params: \u001b[0m\u001b[38;5;34m11\u001b[0m (44.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "data": {
      "text/html": [
       "<pre style=\"white-space:pre;overflow-x:auto;line-height:normal;font-family:Menlo,'DejaVu Sans Mono',consolas,'Courier New',monospace\"><span style=\"font-weight: bold\"> Non-trainable params: </span><span style=\"color: #00af00; text-decoration-color: #00af00\">0</span> (0.00 B)\n",
       "</pre>\n"
      ],
      "text/plain": [
       "\u001b[1m Non-trainable params: \u001b[0m\u001b[38;5;34m0\u001b[0m (0.00 B)\n"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    }
   ],
   "source": [
    "model.summary()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "cbd5c308-de1f-467d-af93-a79758ed1b02",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "WARNING:absl:You are saving your model as an HDF5 file via `model.save()` or `keras.saving.save_model(model)`. This file format is considered legacy. We recommend using instead the native Keras format, e.g. `model.save('my_model.keras')` or `keras.saving.save_model(model, 'my_model.keras')`. \n"
     ]
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved as 'one_layer_model.h5'\n"
     ]
    }
   ],
   "source": [
    "# Save the model\n",
    "model.save('one_layer_model.h5')\n",
    "\n",
    "print(\"Model saved as 'one_layer_model.h5'\")"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "1ecf6774-8063-4866-903d-6e5ef7119db7",
   "metadata": {},
   "source": [
    "## Pump model "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "b261832e-b287-4161-b4cc-238cd4496880",
   "metadata": {},
   "outputs": [],
   "source": [
    "model_path = \"C:\\\\Users\\\\ALHARBR\\\\Downloads\\\\conv_file\\\\model_pump.hdf5\""
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 43,
   "id": "bf19b963-ae0a-4578-b4ea-0ff80789d341",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Check if the file exists before loading\n",
    "if os.path.isfile(model_path):\n",
    "    # Load the Keras model\n",
    "    keras_model = load_model(model_path)\n",
    "    converter = tf.lite.TFLiteConverter.from_keras_model(keras_model)\n",
    "    tflite_model = converter.convert()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 44,
   "id": "9d2217df-d6cc-47c6-87ed-e9104d3e9afd",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Save the converted model as a .tflite file\n",
    "with open('pump_model.tflite', 'wb') as f:\n",
    "    f.write(tflite_model)"
   ]
  },
  {
   "cell_type": "markdown",
   "id": "8aa906e0-b348-4528-b61d-22bde581b5ac",
   "metadata": {},
   "source": [
    "## inference"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 47,
   "id": "31cc1538-02b7-40ee-b26f-1f0175e706a5",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "[[-0.43844798]]\n"
     ]
    }
   ],
   "source": [
    "import tensorflow as tf\n",
    "import numpy as np\n",
    " \n",
    "# Load and run the TFLite model\n",
    "interpreter = tf.lite.Interpreter(model_path=\"one_layer_model.tflite\")\n",
    "interpreter.allocate_tensors()\n",
    " \n",
    "# Create a random input tensor based on the model's input shape\n",
    "input_details = interpreter.get_input_details()\n",
    "input_data = np.random.random_sample(input_details[0]['shape']).astype(np.float32)\n",
    " \n",
    "# Run inference\n",
    "interpreter.set_tensor(input_details[0]['index'], input_data)\n",
    "interpreter.invoke()\n",
    " \n",
    "# Get and print the output\n",
    "output_details = interpreter.get_output_details()\n",
    "output_data = interpreter.get_tensor(output_details[0]['index'])\n",
    "print(output_data)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "b1ac6651-60eb-4691-991b-70d113ef4769",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
