CC = gcc
CFLAGS = -O2 -Wall
LFLAGS = -pthread

all:	pthread_simple pthread_init pthread_join pthread_stack \
	pthread_mutex pthread_mutex2 pthread_papi pthread_cond


pthread_cond:	pthread_cond.o
	$(CC) $(LFLAGS) -o pthread_cond pthread_cond.o

pthread_cond.o:	pthread_cond.c
	$(CC) $(CFLAGS) -c pthread_cond.c

pthread_simple:	pthread_simple.o
	$(CC) $(LFLAGS) -o pthread_simple pthread_simple.o

pthread_simple.o:	pthread_simple.c
	$(CC) $(CFLAGS) -c pthread_simple.c

pthread_init:	pthread_init.o
	$(CC) $(LFLAGS) -o pthread_init pthread_init.o

pthread_init.o:	pthread_init.c
	$(CC) $(CFLAGS) -c pthread_init.c

pthread_join:	pthread_join.o
	$(CC) $(LFLAGS) -o pthread_join pthread_join.o

pthread_join.o:	pthread_join.c
	$(CC) $(CFLAGS) -c pthread_join.c

pthread_mutex:	pthread_mutex.o
	$(CC) $(LFLAGS) -o pthread_mutex pthread_mutex.o

pthread_mutex.o:	pthread_mutex.c
	$(CC) $(CFLAGS) -c pthread_mutex.c

pthread_mutex2:	pthread_mutex2.o
	$(CC) $(LFLAGS) -o pthread_mutex2 pthread_mutex2.o

pthread_mutex2.o:	pthread_mutex2.c
	$(CC) $(CFLAGS) -c pthread_mutex2.c

pthread_stack:	pthread_stack.o
	$(CC) $(LFLAGS) -o pthread_stack pthread_stack.o

pthread_stack.o:	pthread_stack.c
	$(CC) $(CFLAGS) -c pthread_stack.c

pthread_papi:	pthread_papi.o
	$(CC) $(LFLAGS) -o pthread_papi pthread_papi.o -lpapi

pthread_papi.o:	pthread_papi.c
	$(CC) $(CFLAGS) -c pthread_papi.c

clean:	
	rm -f *~ *.o pthread_simple pthread_init pthread_join \
		pthread_stack pthread_mutex pthread_mutex2 pthread_papi \
		pthread_cond
