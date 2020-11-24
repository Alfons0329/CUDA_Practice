#---------------------------Required files-----------------------#
# Thread dim in 2D
TARGET=gb.o
SOURCE=main_thread_2D.cu

# Thread dim in 2D
TARGET_mallocHost=gb_mallocHost.o
SOURCE_mallocHost=main_thread_2D_mallocHost.cu

#------------------------Compiler and flag-----------------------#
# Compilers
CC=g++
NVCC=nvcc

# Libraries and flags
PTHREAD_LIBS=-pthread 
OPENCL_LIBS=-lOpenCL
OPENCV_LIBS=`pkg-config --cflags --libs opencv`
CXXFLAGS= -O2 -std=c++14

#---------------------------Rules-------------------------------#
all: cuda cuda_mallocHost

cuda: $(SOURCE)
	$(NVCC) $(SOURCE) -o $(TARGET) $(CXXFLAGS) $(LIBS) $(OPENCV_LIBS)

cuda_mallocHost: $(SOURCE)
	$(NVCC) $(SOURCE_mallocHost) -o $(TARGET_mallocHost) $(CXXFLAGS) $(LIBS) $(OPENCV_LIBS)

.PHONY: clean

clean:
	rm -f *.o
