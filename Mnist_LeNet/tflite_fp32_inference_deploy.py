# -*- coding: utf-8 -*-
#import tensorflow as tf
import tflite_runtime.interpreter as tflite
import numpy as np

# Configuration
need_convert = False
model_path = "./tflite/converted_model.tflite"

# Run inference
# Load TFLite model and allocate tensors.
num_threads = 4
interpreter = tflite.Interpreter(model_path=model_path, num_threads=num_threads)

# import tensorflow as tf
# interpreter = tf.lite.Interpreter(model_path=model_path)

interpreter.allocate_tensors()

# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Test model on random input data.
input_shape = input_details[0]['shape']

import time
count = 0

while 1:
  input_data = np.array(np.random.random_sample(input_shape), dtype=np.float32)

  interpreter.set_tensor(input_details[0]['index'], input_data)

  start = time.time()

  interpreter.invoke()

  print("iteration {0}, Run time is {1} sec".format(count, time.time() - start))

  # The function `get_tensor()` returns a copy of the tensor data.
  # Use `tensor()` in order to get a pointer to the tensor.
  output_data = interpreter.get_tensor(output_details[0]['index'])
  print(output_data.shape)
  count += 1
  if count >= 10:
    break
