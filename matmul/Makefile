TARGET=main
OBJECTS=util.o matmul.o main.o

CPPFLAGS=-std=c++11 -O3 -Wall -march=znver2 -mavx2 -mno-avx512f -mfma -fopenmp
CPPFLAGS+= -I/usr/local/cuda/include/
LDFLAGS=-lm -lcudart -lcublas -lnvToolsExt
LDFLAGS+=-L/usr/local/cuda/lib64
LDLIBS=-lm -lmpi -lmpi_cxx -lnuma

NVCC=/usr/local/cuda/bin/nvcc

all: $(TARGET)

$(TARGET): $(OBJECTS)
	g++ $(CPPFLAGS) $^ -o $@ $(LDFLAGS)

matmul.o: matmul.cu
	$(NVCC) -c -o $@ $^

clean:
	rm -rf $(TARGET) $(OBJECTS)
