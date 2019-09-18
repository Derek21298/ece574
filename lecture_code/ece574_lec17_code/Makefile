CC = gcc
CFLAGS = -Wall -O2
LFLASG =

all:	hello_world simple_add simple_hello vector_add \
	saxpy saxpy_block saxpy_c


saxpy_c.o:	saxpy_c.c
	$(CC) $(CFLAGS) -c saxpy_c.c


saxpy_c:	saxpy_c.o
	$(CC) $(LFLAGS) -o saxpy_c saxpy_c.o


saxpy:	saxpy.cu
	nvcc -o saxpy saxpy.cu

saxpy_block:	saxpy_block.cu
	nvcc -o saxpy_block saxpy_block.cu



hello_world:	hello_world.cu
	nvcc -o hello_world hello_world.cu


simple_add:	simple_add.cu
	nvcc -o simple_add simple_add.cu


simple_hello:	simple_hello.cu
	nvcc -o simple_hello simple_hello.cu

vector_add:	vector_add.cu
	nvcc -o vector_add vector_add.cu



clean:	
	rm -f *~ *.o simple_add simple_hello hello_world vector_add \
		saxpy saxpy_block saxpy_c

