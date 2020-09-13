## Dataset download
数据集下载地址：http://yann.lecun.com/exdb/mnist/
```buildoutcfg
mkdir dataset
cd dataset
wget http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
wget http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
wget ##下载两个test集合
gzip -d ##解压4个文件
```

## 卷积运算
https://blog.csdn.net/qq_32846595/article/details/79053277
补0的规则
多通道的情况下
https://www.cnblogs.com/lizheng114/p/7498328.html
一张图片通过多个卷积核的计算，可以理解为产生了多个通道

## tf.nn.relu激活函数
https://blog.csdn.net/m0_37870649/article/details/80963053

## tf.nn.max_pool最大值池化函数
https://blog.csdn.net/mzpmzk/article/details/78636184

## Profile
```buildoutcfg
/home/lesliefang/tflite/tensorflow/tensorflow-2.3.0/bazel-bin/tensorflow/lite/tools/benchmark/benchmark_model --graph=/home/lesliefang/tflite/mnist/Mnist_LeNet/tflite/converted_model.tflite --num_threads=4
```