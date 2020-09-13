## 介绍
tfllite 有两个部分组成:

convert: 用于将模型转换成tflite格式,可以从keras模型，也可以从各种tf1.x的模型转换出来(比如save model, pb model)
interpreter: tflite 运行时，调试的时候可以直接用集成在tf里面，部署的时候有单独的whl包作为更小的一个运行时(https://www.tensorflow.org/lite/guide/python)

## 转换模型
推荐用SavedModel的格式
https://www.tensorflow.org/lite/guide/get_started#2_convert_the_model_format
https://www.tensorflow.org/lite/convert

TF1.x
https://www.tensorflow.org/api_docs/python/tf/compat/v1/lite/TFLiteConverter#from_frozen_graph
https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/r1/convert/python_api.md#checkpoints


## 如何单独编译tflite的运行时
```
# only build C++ so
bazel --output_user_root=$build_dir build -c opt //tensorflow/lite:tensorflowlite


# Build whl package
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/pip_package
# install dependecy and run script
yum install swig libjpeg-devel

pip install numpy pybind11
tensorflow/lite/tools/make/download_dependencies.sh
tensorflow/lite/tools/pip_package/build_pip_package.sh

看build_pip_package.sh 的输出日志或者
在代码目录下搜索：tflite_runtime-2.1.0-cp36-cp36m-linux_x86_64.whl
可以找到对应的生成的whl package
```
一个tflite模型可以被多个解释器调用，每个解释器需要独占一个线程(多线程不能共享解释器)


## Profile
https://www.tensorflow.org/lite/guide/faq 最下面提到
https://www.tensorflow.org/lite/performance/measurement
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark

1. 编译benchmark model
```
bazel build -c opt //tensorflow/lite/tools/benchmark:benchmark_model
```
2. 使用benchmark Model
```
bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model   --graph=your_model.tflite --num_threads=4
```
benchmark的参数
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark#parameters

例子
```
/home/lesliefang/tflite/tensorflow/tensorflow-2.3.0/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model --graph=/home/lesliefang/tflite/mnist/Mnist_LeNet/tflite/converted_model.tflite --num_threads=4 --enable_op_profiling=true --profiling_output_csv_file=/root/fp32.csv --verbose
```
profileing 的结果放到 /root/fp32.csv 文件里面

## MLIR
https://www.tensorflow.org/mlir
https://mlir.llvm.org/

## 添加算子
https://www.tensorflow.org/lite/guide/inference#write_a_custom_operator
https://www.tensorflow.org/lite/guide/ops_custom

好像没有单个op的benchmark工具，要测试单个op的性能，需要创建只包含这个op的模型，然后使用benchmark工具

## 写C++ 应用
用bazel的话，参考这个
https://www.tensorflow.org/lite/guide/inference#load_and_run_a_model_in_c

用Makefile参考 tensorflow/lite/tools/make/Makefile
里面会去build一个label_image的例子
https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/examples/label_image

即使用bazel build的话，也会生成tflite的静态链接库
libtensorflow-lite.a

C++ API的地址
https://www.tensorflow.org/lite/api_docs/cc/class/tflite/impl/interpreter#classtflite_1_1impl_1_1_interpreter_1a259df676187dc3fb312d4b7e3417a7de

## Tflite+SGX
https://github.com/occlum/occlum/tree/master/demos/tensorflow_lite
### eigen gzip报错(eigen 没有下载全)
设置https代理成http代理
