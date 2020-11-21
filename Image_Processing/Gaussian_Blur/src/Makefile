#---------------------------Required files-----------------------#
# Thread dim in 1D
TARGET_1D=gb_1D.o
SOURCE_1D=main_thread_1D.cu

#------------------------Compiler and flag-----------------------#
# Compilers
CC=g++
NVCC=nvcc

# Libraries and flags
PTHREAD_LIBS=-pthread 
OPENCL_LIBS=-lOpenCL
OPENCV_LIBS=`pkg-config --cflags --libs opencv`
CXXFLAGS=-O3 -std=c++14

#---------------------------Rules-------------------------------#
all: cuda_1D

# Rules for pthread Gaussian Blur
pthread: $(SOURCE_PTHREAD)
	$(CC) $(SOURCE_PTHREAD) -o $(TARGET_PTHREAD) $(CXXFLAGS) $(PTHREAD_LIBS) $(LIBS) $(OPENCV_LIBS)

 #Rules for CUDA Gaussian Blur
cuda_1D: $(SOURCE_1D)
	$(NVCC) $(SOURCE_1D) -o $(TARGET_1D) $(CXXFLAGS) $(LIBS) $(OPENCV_LIBS)

.PHONY: clean

clean:
	rm -f *.o