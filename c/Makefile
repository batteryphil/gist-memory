CC ?= gcc
CFLAGS ?= -O3 -Wall -Wextra -std=c99 -pedantic

all: test_gist_c

test_gist_c: test_gist_c.c gist_memory.h
	$(CC) $(CFLAGS) test_gist_c.c -o test_gist_c -lm

test: test_gist_c
	./test_gist_c

clean:
	rm -f test_gist_c *.o

.PHONY: all test clean
