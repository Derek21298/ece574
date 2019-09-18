CC = gcc
CFLAGS = -O2 -Wall -g -fopenmp
LFLAGS = -ljpeg -lm -lpapi -fopenmp

all:   sobel_before sobel_coarse sobel_fine sobel_cool

sobel_before:	sobel_before.o
		$(CC) -o sobel_before sobel_before.o $(LFLAGS)

sobel_before.o:	sobel_before.c
		$(CC) $(CFLAGS) -c sobel_before.c


sobel_coarse:	sobel_coarse.o
		$(CC) -o sobel_coarse sobel_coarse.o $(LFLAGS)

sobel_coarse.o:	sobel_coarse.c
		$(CC) $(CFLAGS) -c sobel_coarse.c


sobel_cool:	sobel_cool.o
		$(CC) -o sobel_cool sobel_cool.o $(LFLAGS)

sobel_cool.o:	sobel_cool.c
		$(CC) $(CFLAGS) -c sobel_cool.c



sobel_fine:	sobel_fine.o
		$(CC) -o sobel_fine sobel_fine.o $(LFLAGS)

sobel_fine.o:	sobel_fine.c
		$(CC) $(CFLAGS) -c sobel_fine.c

clean:
	       rm -f *~ *.o sobel_coarse sobel_fine sobel_before sobel_cool out.jpg


submit:
	tar -czvf hw05_submit.tar.gz Makefile README *.c

