#include <iostream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <fstream>
#include "absl/memory/memory.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/optional_debug_tools.h"
#include "tensorflow/lite/profiling/profiler.h"
#include "dataset_helper.h"

#define LOG(x) std::cerr
double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

void test(){
  //int batchsize = 128;
  std::string model_name = "../tflite/converted_model.tflite";
  //std::string model_name = "../tflite/converted_int8_model.tflite";

  std::unique_ptr<tflite::FlatBufferModel> model;
  std::unique_ptr<tflite::Interpreter> interpreter;
  model = tflite::FlatBufferModel::BuildFromFile(model_name.c_str());
  if (!model) {
    LOG(FATAL) << "\nFailed to mmap model " << model_name << "\n";
    exit(-1);
  }
  tflite::FlatBufferModel* f_model = model.get();
  LOG(INFO) << "Loaded model " << model_name << "\n";
  model->error_reporter();
  LOG(INFO) << "resolved reporter\n";

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*model, resolver)(&interpreter);
  if (!interpreter) {
    LOG(FATAL) << "Failed to construct interpreter\n";
    exit(-1);
  }
  bool verbose = false;
  if (verbose) {
    LOG(INFO) << "tensors size: " << interpreter->tensors_size() << "\n";
    LOG(INFO) << "nodes size: " << interpreter->nodes_size() << "\n";
    LOG(INFO) << "inputs: " << interpreter->inputs().size() << "\n";
    LOG(INFO) << "input(0) name: " << interpreter->GetInputName(0) << "\n";

    int t_size = interpreter->tensors_size();
    for (int i = 0; i < t_size; i++) {
      if (interpreter->tensor(i)->name)
        LOG(INFO) << i << ": " << interpreter->tensor(i)->name << ", "
                  << interpreter->tensor(i)->bytes << ", "
                  << interpreter->tensor(i)->type << ", "
                  << interpreter->tensor(i)->params.scale << ", "
                  << interpreter->tensor(i)->params.zero_point << "\n";
    }
  }

  int number_of_threads = 4;
  interpreter->SetNumThreads(number_of_threads); //when create interpreter, we can also set the threads_number

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();
  if (verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

  // interpreter->inputs() 拿到所有input的索引存储的list,每个索引值参考tflite模型input的location属性，不是0或者1，是在convert 模型的时候加上去的，比如在这个例子的模型里面，inputs的第一个位置（位置0)拿到的索引是0，outputs的第一个位置（位置0)拿到的索引是16
  // 我理解这个index，应该是把模型的所有tensor都编了个索引，后面通过interpreter->tensor 或者 interpreter->typed_tensor 等方法接上这个索引号就可以拿到对应的tensor了, inputs 和 outputs 是特例
  int input = inputs[0];
  int output = outputs[0];

  LOG(INFO) << "input position 0's index is : " << input << "\n";
  LOG(INFO) << "output position 0's index is : " << output << "\n";

  //Get the input shape
  TfLiteIntArray* input_dims = interpreter->tensor(input)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  auto batchsize = input_dims->data[0]; //Get the BS from the model shape, which will be used in allcator tensor memory
  auto input_size = input_dims->data[input_dims->size - 1];

  //Get the output shape
  TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
  // assume output dims to be something like (1, 1, ... ,size)
  // auto batchsize = output_dims->data[0]; //Same as input batchsize
  auto output_size = output_dims->data[output_dims->size - 1];

  LOG(INFO) << "batchsize is : " << batchsize << "\n"; //128
  LOG(INFO) << "input_size is : " << input_size << "\n"; //784
  LOG(INFO) << "output_size is : " << output_size << "\n"; //10

  if (interpreter->AllocateTensors() != kTfLiteOk) {
    LOG(FATAL) << "Failed to allocate tensors!";
  }

  auto input_data = interpreter->typed_tensor<float>(input);

  std::string inference_image_path = "../dataset/t10k-images-idx3-ubyte";
  std::string inference_label_path = "../dataset/t10k-labels-idx1-ubyte";
  inference_images* images = new inference_images(inference_image_path);
  inference_labels* labels = new inference_labels(inference_label_path);

  std::vector<std::vector<float> > input_images;
  std::vector<float> input_labels;

  int images_number = images->get_images_number();
  //std::cout<<images_number<<std::endl;

  int loop_count = images_number/batchsize;
  //loop_count = 1;

  struct timeval start_time, stop_time;
  int iteration = 0;
  int correct_count = 0;

  for (int i = 0; i < loop_count; i++) {
    input_images = images->read_images(batchsize);
    input_labels = labels->read_labels(batchsize);

    // Copy the data into input tensor
    for(int bs = 0;bs<batchsize;bs++){
      for(int pixel=0;pixel<784;pixel++){
          input_data[bs*784+pixel] = input_images[bs][pixel];
      }
    }

    gettimeofday(&start_time, nullptr);

    if (interpreter->Invoke() != kTfLiteOk) {
      LOG(FATAL) << "Failed to invoke tflite!\n";
    }

    gettimeofday(&stop_time, nullptr);
    LOG(INFO) << "invoke iteration: " << iteration <<"\n";
    LOG(INFO) << "average time: "
              << (get_us(stop_time) - get_us(start_time)) / (1000)
              << " ms \n";

    float * res = interpreter->typed_output_tensor<float>(0);

    for(int i=0;i<batchsize;i++){
      int digit = 0;
      float max = res[i*10];
      for(int j=0;j<10;j++){
        // print all 10 digit's possibility
        if(res[i*10+j]>max){
            max = res[i*10+j];
            digit = j;
        }
      }
      // LOG(INFO) << "Input digit is: " << digit << "\n";
      // LOG(INFO) << "input_labels is: " << input_labels[i] << "\n";
      if(digit == input_labels[i]){
         correct_count += 1;
      }

    }
     iteration += 1;
  }
  LOG(INFO) << "Correct Count is: " << correct_count << "\n";
  LOG(INFO) << "Total Images is: " << loop_count*batchsize << "\n";
  LOG(INFO) << "Accuracy is: " << correct_count/float(loop_count*batchsize) << "\n";

  delete images;
  delete labels;

  return;
}

int main(int argc, char ** argv){
    //td::cout<<"Hello World"<<std::endl;
    test();
//    dataset_help();
    return 0;
}
