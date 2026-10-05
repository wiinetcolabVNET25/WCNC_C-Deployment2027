CC = gcc
CFLAGS = -O2 -Wall -Wextra -std=c11
LDFLAGS = -lm

TARGET = SA_deployment
OTHERS = config.h
SOURCES = main.c trace.c grid.c metric.c sa.c

$(TARGET): $(SOURCES) $(OTHERS)
	$(CC) $(CFLAGS) $(SOURCES) -o $(TARGET) $(LDFLAGS)

clean:
	rm -f $(TARGET)