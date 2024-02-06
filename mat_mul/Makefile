TARGET=main
OBJECTS=util.o mat_mul.o

CPPFLAGS=-std=c++11 -Wall -O3 -march=znver2 -fopenmp -fcf-protection=none -fno-stack-protector -ftree-vectorize -funroll-loops -ffast-math
LDFLAGS=-lm -pthread

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
