#ifndef TENSORFLOW_LITE_EXAMPLES_MNIST_LESLIE_H_
#define TENSORFLOW_LITE_EXAMPLES_MNIST_LESLIE_H_

class inference_images{
public:
    inference_images(std::string trainDataFile);
    ~inference_images();
    std::vector<std::vector<float> > read_images(int batchsize);
    int get_images_number();
private:
	int32_t images_numbers;
	int32_t magic_number;
	int32_t row_number;
	int32_t column_number;
	std::ifstream * inFile;
	void read_prefix();
};

class inference_labels{
public:
    inference_labels(std::string trainDataFile);
    ~inference_labels();
    std::vector<float> read_labels(int batchsize);
private:
	int images_numbers;
	int magic_number;
    std::ifstream * inFile;
	void read_prefix();
};

//void dataset_help();

#endif