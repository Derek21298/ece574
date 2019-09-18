CC = gcc
CFLAGS = -O2 -Wall -g
LFLAGS = -ljpeg -lm -lpapi -lpthread

all:   sobel_coarse sobel_fine

sobel_coarse:	sobel_coarse.o
		$(CC) -o sobel_coarse sobel_coarse.o $(LFLAGS)

sobel_coarse.o:	sobel_coarse.c
		$(CC) $(CFLAGS) -c sobel_coarse.c


sobel_fine:	sobel_fine.o
		$(CC) -o sobel_fine sobel_fine.o $(LFLAGS)

sobel_fine.o:	sobel_fine.c
		$(CC) $(CFLAGS) -c sobel_fine.c



clean:
	       rm -f *~ *.o sobel_coarse sobel_fine

submit:
	tar -czvf hw04_submit.tar.gz Makefile README *.c

