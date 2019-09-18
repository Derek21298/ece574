CC = gcc
CFLAGS = -O2
LFLAGS = -ljpeg -lm -lpapi -lOpenCL

all:	sobel_opencl

sobel_opencl:	sobel_opencl.c
	$(CC) $(CFLAGS) $(LFLAGS) -o sobel_opencl  sobel_opencl.c

clean:
	rm -f *~ *.o sobel_opencl out.jpg

submit:
	tar -czvf hw09_submit.tar.gz Makefile README *.c



