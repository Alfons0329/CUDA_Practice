#---------------------------Required files-----------------------#
# Thread dim in 1D
TARGET_1D=gb_1D.o
SOURCE_1D=main_thread_1D.cu

# Thread dim in 2D
TARGET_2D=gb_2D.o
SOURCE_2D=main_thread_2D.cu

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
all: cuda_1D cuda_2D

pthread: $(SOURCE_PTHREAD)
	$(CC) $(SOURCE_PTHREAD) -o $(TARGET_PTHREAD) $(CXXFLAGS) $(PTHREAD_LIBS) $(LIBS) $(OPENCV_LIBS)

cuda_1D: $(SOURCE_1D)
	$(NVCC) $(SOURCE_1D) -o $(TARGET_1D) $(CXXFLAGS) $(LIBS) $(OPENCV_LIBS)

cuda_2D: $(SOURCE_2D)
	$(NVCC) $(SOURCE_2D) -o $(TARGET_2D) $(CXXFLAGS) $(LIBS) $(OPENCV_LIBS)

.PHONY: clean

clean:
	rm -f *.o
