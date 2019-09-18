NVCC = nvcc
CC = gcc
CFLAGS = -O2
LFLAGS = -ljpeg -lm -lpapi

all:	sobel_before sobel_coarse sobel_fine

sobel_before:	sobel_before.o
	$(CC) $(LFLAGS) -o sobel_before sobel_before.o

sobel_before.o:	sobel_before.c
	$(CC) $(CFLAGS) -c sobel_before.c

sobel_coarse:	sobel_coarse.cu
	$(NVCC) $(CFLAGS) $(LFLAGS) -o sobel_coarse  sobel_coarse.cu

sobel_fine:	sobel_fine.cu
	$(NVCC) $(CFLAGS) $(LFLAGS) -o sobel_fine  sobel_fine.cu

clean:
	rm -f *~ *.o sobel_coarse sobel_before sobel_fine out.jpg

submit:
	tar -czvf hw08_submit.tar.gz Makefile README *.c *.cu



