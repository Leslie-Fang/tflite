#include <fstream>
#include <string>
#include <iostream>
#include <vector>
#include "dataset_helper.h"

inference_images::inference_images(std::string trainDataFile){
    inFile = new std::ifstream(trainDataFile, std::ios::in|std::ios::binary);
    read_prefix();
}

inference_images::~inference_images(){
   inFile->close();
   delete inFile;
}

int inference_images::get_images_number(){
   return images_numbers;
}

void inference_images::read_prefix(){
    char ptr[4];
    inFile->read(ptr, 4);
    magic_number = (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3]; //big endien storage
    std::cout<<"inference_images magic_number:" << magic_number<<std::endl;

    inFile->read(ptr, 4);
    images_numbers = (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
    std::cout<<"inference_images image_number:" << images_numbers<<std::endl;

    inFile->read(ptr, 4);
    row_number = (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
    std::cout<<"inference_images row_number:" << row_number<<std::endl;

    inFile->read(ptr, 4);
    column_number = (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
    std::cout<<"inference_images column_number:" << column_number<<std::endl;

    return;
}

std::vector<std::vector<float> > inference_images::read_images(int batchsize){
    std::vector<std::vector<float> > images_pix_float;
    char ptr[1];
    for(int i=0;i<batchsize;i++){
        std::vector<float> image_pix_float;
        for(int j=0;j<row_number*column_number;j++){
            inFile->read(ptr, 1);
            //int8_t temp = ptr[0];
            //std::cout<<float(ptr[0])<<std::endl;
            image_pix_float.push_back(uint8_t(ptr[0])/255.0);
        }
        images_pix_float.push_back(image_pix_float);
    }
    return images_pix_float;
}

inference_labels::inference_labels(std::string trainDataFile){
    inFile = new std::ifstream(trainDataFile, std::ios::in|std::ios::binary);
    read_prefix();
}

inference_labels::~inference_labels(){
   inFile->close();
   delete inFile;
}

void inference_labels::read_prefix(){
    char ptr[4];
    inFile->read(ptr, 4);
    magic_number = (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3]; //big endien storage
    std::cout<<"inference_images magic_number:" << magic_number<<std::endl;

    inFile->read(ptr, 4);
    images_numbers = (ptr[0] << 24) | (ptr[1] << 16) | (ptr[2] << 8) | ptr[3];
    std::cout<<"inference_images image_number:" << images_numbers<<std::endl;

    return;
}

std::vector<float> inference_labels::read_labels(int batchsize){
    std::vector<float> labels;
    char ptr[1];
    for(int i=0;i<batchsize;i++){
//        std::vector<float> image_pix_float;
//        for(int j=0;j<row_number*column_number;j++){
          inFile->read(ptr, 1);
            //int8_t temp = ptr[0];
            //std::cout<<float(ptr[0])<<std::endl;
          //std::cout<<float(ptr[0])<<std::endl;
          labels.push_back(float(ptr[0]));
//        }
//        images_pix_float.push_back(image_pix_float);
    }
    return labels;
}

//void dataset_help(int batchsize){
//    std::string inference_image_path = "../dataset/t10k-images-idx3-ubyte";
//    std::string inference_label_path = "../dataset/t10k-labels-idx1-ubyte";
//    inference_images* images = new inference_images(inference_image_path);
//    inference_labels* labels = new inference_labels(inference_label_path);
//
//    images->read_images(1);
//    labels->read_labels(128);
//
//    delete images;
//    delete labels;
//
//    return;
//}