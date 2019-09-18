CC = gcc
CFLAGS = -O2 -Wall -fopenmp
LFLAGS = -fopenmp

all:	openmp_simple openmp_for \
	openmp_clauses openmp_reduction openmp_section \
	openmp_static_schedule openmp_dynamic_schedule \
	openmp_critical openmp_simd openmp_offload \
	openmp_simd_reduction openmp_dynamic_chunk


openmp_critical:	openmp_critical.o
	$(CC) $(LFLAGS) -o openmp_critical openmp_critical.o

openmp_critical.o:	openmp_critical.c
	$(CC) $(CFLAGS) -c openmp_critical.c

openmp_offload:	openmp_offload.o
	$(CC) $(LFLAGS) -o openmp_offload openmp_offload.o

openmp_offload.o:	openmp_offload.c
	$(CC) $(CFLAGS) -c openmp_offload.c


openmp_section:	openmp_section.o
	$(CC) $(LFLAGS) -o openmp_section openmp_section.o

openmp_section.o:	openmp_section.c
	$(CC) $(CFLAGS) -c openmp_section.c

openmp_simd:	openmp_simd.o
	$(CC) $(LFLAGS) -o openmp_simd openmp_simd.o

openmp_simd.o:	openmp_simd.c
	$(CC) $(CFLAGS) -c openmp_simd.c


openmp_simple:	openmp_simple.o
	$(CC) $(LFLAGS) -o openmp_simple openmp_simple.o

openmp_simple.o:	openmp_simple.c
	$(CC) $(CFLAGS) -c openmp_simple.c

openmp_for:	openmp_for.o
	$(CC) $(LFLAGS) -o openmp_for openmp_for.o

openmp_for.o:	openmp_for.c
	$(CC) $(CFLAGS) -c openmp_for.c

openmp_join:	openmp_join.o
	$(CC) $(LFLAGS) -o openmp_join openmp_join.o

openmp_join.o:	openmp_join.c
	$(CC) $(CFLAGS) -c openmp_join.c


openmp_reduction:	openmp_reduction.o
	$(CC) $(LFLAGS) -o openmp_reduction openmp_reduction.o

openmp_reduction.o:	openmp_reduction.c
	$(CC) $(CFLAGS) -c openmp_reduction.c


openmp_simd_reduction:	openmp_simd_reduction.o
	$(CC) $(LFLAGS) -o openmp_simd_reduction openmp_simd_reduction.o

openmp_simd_reduction.o:	openmp_simd_reduction.c
	$(CC) $(CFLAGS) -c openmp_simd_reduction.c


openmp_papi:	openmp_papi.o
	$(CC) $(LFLAGS) -o openmp_papi openmp_papi.o -lpapi

openmp_papi.o:	openmp_papi.c
	$(CC) $(CFLAGS) -c openmp_papi.c

openmp_clauses:	openmp_clauses.o
	$(CC) $(LFLAGS) -o openmp_clauses openmp_clauses.o

openmp_clauses.o:	openmp_clauses.c
	$(CC) $(CFLAGS) -c openmp_clauses.c


openmp_static_schedule:	openmp_static_schedule.o
	$(CC) $(LFLAGS) -o openmp_static_schedule openmp_static_schedule.o

openmp_static_schedule.o:	openmp_static_schedule.c
	$(CC) $(CFLAGS) -c openmp_static_schedule.c

openmp_dynamic_schedule:	openmp_dynamic_schedule.o
	$(CC) $(LFLAGS) -o openmp_dynamic_schedule openmp_dynamic_schedule.o

openmp_dynamic_schedule.o:	openmp_dynamic_schedule.c
	$(CC) $(CFLAGS) -c openmp_dynamic_schedule.c


openmp_dynamic_chunk:	openmp_dynamic_chunk.o
	$(CC) $(LFLAGS) -o openmp_dynamic_chunk openmp_dynamic_chunk.o

openmp_dynamic_chunk.o:	openmp_dynamic_chunk.c
	$(CC) $(CFLAGS) -c openmp_dynamic_chunk.c



clean:	
	rm -f *~ *.o openmp_simple openmp_for openmp_join \
		openmp_reduction openmp_mutex openmp_mutex2 openmp_papi \
		openmp_clauses openmp_section \
		openmp_static_schedule openmp_dynamic_schedule \
		openmp_critical openmp_simd openmp_offload \
		openmp_simd_reduction openmp_dynamic_chunk
