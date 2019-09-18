CC = gcc
CFLAGS = -O2 -Wall -g
LFLAGS = -lpapi -ljpeg -lm

all:   sobel sobel_improved


sobel:	sobel.o
		$(CC) -o sobel sobel.o $(LFLAGS)

sobel.o:	sobel.c
		$(CC) $(CFLAGS) -c sobel.c

sobel_improved:	sobel_improved.o
		$(CC) -o sobel_improved sobel_improved.o $(LFLAGS)

sobel_improved.o:	sobel_improved.c
		$(CC) $(CFLAGS) -c sobel_improved.c


clean:
	       rm -f *~ *.o sobel sobel_improved


submit:	
	tar -czvf hw03_submit.tar.gz Makefile README *.c
