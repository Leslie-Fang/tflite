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
  int batchsize = 128;
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
  interpreter->SetNumThreads(number_of_threads);

  int input = interpreter->inputs()[0];
  if (verbose) LOG(INFO) << "input: " << input << "\n";

  const std::vector<int> inputs = interpreter->inputs();
  const std::vector<int> outputs = interpreter->outputs();
  if (verbose) {
    LOG(INFO) << "number of inputs: " << inputs.size() << "\n";
    LOG(INFO) << "number of outputs: " << outputs.size() << "\n";
  }

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
  std::cout<<images_number<<std::endl;

  int loop_count = images_number/batchsize;
  //loop_count = 1;

  struct timeval start_time, stop_time;
  int iteration = 0;
  int correct_count = 0;

  for (int i = 0; i < loop_count; i++) {
    input_images = images->read_images(batchsize);
    input_labels = labels->read_labels(batchsize);

    for(int bs = 0;bs<batchsize;bs++){
      for(int pixel=0;pixel<784;pixel++){

          //LOG(INFO) << input_images[bs][pixel] << "\n";
          input_data[bs*784+pixel] = input_images[bs][pixel];
      }
      //LOG(INFO) << "Next Image"<<"\n";
      //break;
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

    int output = interpreter->outputs()[0];
    TfLiteIntArray* output_dims = interpreter->tensor(output)->dims;
    //LOG(INFO) << output_dims->size << "\n"; //2
    // assume output dims to be something like (1, 1, ... ,size)
    auto bb = output_dims->data[0];
    auto output_size = output_dims->data[output_dims->size - 1];

//    LOG(INFO) << bb << "\n"; //128
//    LOG(INFO) << output_size << "\n"; //10

    float * res = interpreter->typed_output_tensor<float>(0);

    for(int i=0;i<batchsize;i++){
      int digit = 0;
      float max = res[i*10];
      for(int j=0;j<10;j++){
        // print all 10 digit's possibility
        // LOG(INFO) << res[i] << "\n";
        if(res[i*10+j]>max){
            max = res[i*10+j];
            digit = j;
        }
      }
//      LOG(INFO) << "Input digit is: " << digit << "\n";
//      LOG(INFO) << "input_labels is: " << input_labels[i] << "\n";
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