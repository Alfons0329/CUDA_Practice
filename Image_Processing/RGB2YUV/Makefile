#---------------------------Required files-----------------------#
SRC=./src/
BIN=./bin/

SOURCE=$(SRC)main.cpp
TARGET=$(BIN)main_out

#------------------------Compiler and flag-----------------------#
# Compilers
CC=g++

# Libraries and flags
OPENCV_LIBS=`pkg-config --cflags --libs opencv`
LDFLAGS=-L../lib64 -L/usr/local/cuda-11.0/lib64
INCLUDES=-I./include -I/usr/local/cuda-11.0/include
LIBS=-ldl -lrt -lcudart -lnvjpeg
RPATH=-Wl,-rpath=../lib64 -Wl,-rpath=/usr/local/cuda-11.0/lib64
CXXFLAGS:= -O2 -std=c++14 $(LDFLAGS) $(INCLUDES) $(LIBS) $(RPATH)

#---------------------------Rules-------------------------------#
all: cuda

cuda: $(SOURCE)  
	$(CC) -m64 $(SOURCE) $(CXXFLAGS) -o $(TARGET)

.PHONY: clean

clean:
	rm -f $(BIN)*.o