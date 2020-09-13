CC=g++
INC=-I/home/lesliefang/tflite/tensorflow/tensorflow-2.3.0 -I/home/lesliefang/tflite/tensorflow/tensorflow-2.3.0/tensorflow/lite/tools/make/downloads/flatbuffers/include -I/home/lesliefang/tflite/tensorflow/tensorflow-2.3.0/tensorflow/lite/tools/make/downloads/absl
STATIC=
LINKLIBS=
SOURCES=main.cpp dataset_helper.cpp
OBJECTS=$(SOURCES:.c=.o)
LIB_PATH := /home/lesliefang/tflite/tensorflow/tensorflow-2.3.0/tensorflow/lite/tools/make/gen/linux_x86_64/lib/libtensorflow-lite.a
LIBS := \
-lstdc++ \
-lpthread \
-lm \
-lz \
-ldl

main : $(OBJECTS) $(LIB_PATH)
	$(CC) $(INC) $(STATIC) $(LINKLIBS) -o $@ $(OBJECTS) $(LIB_PATH) $(LIBS)
clean:
	rm -rf *.o *.d main
