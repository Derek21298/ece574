MPICC = mpicc
CFLAGS = -O2 -Wall
LFLAGS =

all:	mpi_barrier mpi_bcast mpi_test mpi_hello mpi_reduce \
	mpi_scatter mpi_send mpi_wtime mpi_gather

mpi_barrier:	mpi_barrier.o
	$(MPICC) -o mpi_barrier mpi_barrier.o

mpi_barrier.o:	mpi_barrier.c
	$(MPICC) $(CFLAGS) -c mpi_barrier.c

mpi_bcast:	mpi_bcast.o
	$(MPICC) -o mpi_bcast mpi_bcast.o

mpi_bcast.o:	mpi_bcast.c
	$(MPICC) $(CFLAGS) -c mpi_bcast.c

mpi_test:	mpi_test.o
	$(MPICC) -o mpi_test mpi_test.o

mpi_test.o:	mpi_test.c
	$(MPICC) $(CFLAGS) -c mpi_test.c

mpi_gather:	mpi_gather.o
	$(MPICC) -o mpi_gather mpi_gather.o

mpi_gather.o:	mpi_gather.c
	$(MPICC) $(CFLAGS) -c mpi_gather.c


mpi_hello:	mpi_hello.o
	$(MPICC) -o mpi_hello mpi_hello.o

mpi_hello.o:	mpi_hello.c
	$(MPICC) $(CFLAGS) -c mpi_hello.c

mpi_reduce:	mpi_reduce.o
	$(MPICC) -o mpi_reduce mpi_reduce.o

mpi_reduce.o:	mpi_reduce.c
	$(MPICC) $(CFLAGS) -c mpi_reduce.c

mpi_scatter:	mpi_scatter.o
	$(MPICC) -o mpi_scatter mpi_scatter.o

mpi_scatter.o:	mpi_scatter.c
	$(MPICC) $(CFLAGS) -c mpi_scatter.c


mpi_send:	mpi_send.o
	$(MPICC) -o mpi_send mpi_send.o

mpi_send.o:	mpi_send.c
	$(MPICC) $(CFLAGS) -c mpi_send.c


mpi_wtime:	mpi_wtime.o
	$(MPICC) -o mpi_wtime mpi_wtime.o

mpi_wtime.o:	mpi_wtime.c
	$(MPICC) $(CFLAGS) -c mpi_wtime.c



clean:	
	rm -f *~ *.o mpi_barrier mpi_bcast \
		mpi_test mpi_hello mpi_reduce \
		mpi_scatter mpi_send mpi_wtime mpi_gather
