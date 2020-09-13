# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import datetime
import time
import os
from dataset_help import inference_images
from dataset_help import inference_labels

# Configuration
need_convert = False
run_inference = True

if os.path.isdir(os.path.join(os.getcwd(), "tflite")) is False:
  os.makedirs(os.path.join(os.getcwd(), "tflite"))

model_path = "./tflite/converted_model.tflite"

base_path = os.getcwd()
base_inference_path = os.path.join(base_path, "dataset")
inference_image_path = os.path.join(base_inference_path, "t10k-images-idx3-ubyte")
inference_label_path = os.path.join(base_inference_path, "t10k-labels-idx1-ubyte")
inference_label_obj = inference_labels(inference_label_path)
inference_image_obj = inference_images(inference_image_path)

raw_Data = []
raw_Label = []
quan_steps = inference_image_obj.get_images_number()
for i in range(quan_steps):
  raw_Data.append(inference_image_obj.read_images(1)[0])
  raw_Label.append(inference_label_obj.read_labels(1)[0])


batchsize = 128

train = tf.convert_to_tensor(np.array(raw_Data, dtype='float32'))
train_label = tf.convert_to_tensor(np.array(raw_Label, dtype='float32'))
dataset = tf.data.Dataset.from_tensor_slices((train, train_label)).batch(batchsize, drop_remainder=True)

# Convert model into tflite
if need_convert:
  # Path to the frozen graph file
  graph_def_file = './pb_models/freeze_fp32.pb'
  # A list of the names of the model's input tensors
  input_arrays = ['X']
  # A list of the names of the model's output tensors
  output_arrays = ['Ys']
  # Load and convert the frozen graph
  # # not specify the input shape
  # converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
  #   graph_def_file, input_arrays, output_arrays)

  converter = tf.compat.v1.lite.TFLiteConverter.from_frozen_graph(
    graph_def_file = graph_def_file,
    input_arrays = input_arrays,
    input_shapes= {'X': [batchsize, 784]},
    output_arrays = output_arrays)

  tflite_model = converter.convert()
  # Write the converted model to disk
  open(model_path, "wb").write(tflite_model)

if run_inference:
  # Run inference
  # Load TFLite model and allocate tensors.
  num_threads = 4

  interpreter = tf.lite.Interpreter(model_path=model_path, num_threads=num_threads)
  interpreter.allocate_tensors()

  # Get input and output tensors.
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Test model on random input data.
  input_shape = input_details[0]['shape']

  correct = 0
  global_images = 0

  # dataset = dataset.shuffle(buffer_size=100)
  # dataset = dataset.batch(batch_size=128)
  # dataset = dataset.prefetch(1)

  import time

  iteration = 0

  for data in dataset:

    input_data = data[0]
    # print(input_data.size())
    # break

    interpreter.set_tensor(input_details[0]['index'], input_data)

    start = time.time()

    interpreter.invoke()

    print("iteration {0}, Run time is {1} sec".format(iteration, time.time() - start))

    output_data = interpreter.get_tensor(output_details[0]['index'])

    label = data[1]
    # print(type(output_data[0]))
    # print(output_data.shape)
    # if label == np.argmax(output_data[0]):
    #   correct += 1

    for image_number in range(batchsize):
      maxindex = label[image_number]
      true_label = np.argmax(output_data[image_number])
      # print(results.shape)
      # print(inference_y.shape)

      # print("maxindex is {0}".format(maxindex))
      # print("true_label is {0}".format(true_label))
      if maxindex == true_label:
        correct += 1

    iteration += 1
    global_images += batchsize

    #break

    # if global_images >= 1000:
    #   break
  print("correct count is:{}".format(correct))
  print("global_images is :{}".format(global_images))
  print("accuracy is :{0}".format(float(correct)/global_images))