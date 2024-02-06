TARGET=main
OBJECTS=util.o vec_add.o

CC=mpicxx
CPPFLAGS=-std=c++11 -O3 -Wall -march=znver2 -fopenmp -fcf-protection=none -fno-stack-protector
LDFLAGS=-lm -pthread
LDLIBS=-lnuma -lm

all: $(TARGET)

$(TARGET): $(OBJECTS)

clean:
	rm -rf $(TARGET) $(OBJECTS)
