TARGET = main
OBJECTS = util.o communication.o
CC = mpic++
CFLAGS = -Wall

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CC) $(CFLAGS) -o $(TARGET) $(OBJECTS)

%.o: %.c
	$(CC) $(CFLAGS) -c $< -o $@

hostname: hostname.c
error: error.c
send_rece: send_rece.c
communication: communication.o util.o

.PHONY: clean
clean:
	rm -rf $(TARGET) $(OBJECTS)