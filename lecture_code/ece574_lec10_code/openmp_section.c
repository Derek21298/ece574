/* Test openmp section directive */

#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

static unsigned char *memory;

int main (int argc, char **argv) {

	int mem_size=512*1024*1024;	/* 512 MB */
	int i;
	int num_threads=2;

	/* Set number of threads from the command line */
        if (argc>1) {
		num_threads=atoi(argv[1]);
	}

	/* allocate memory */
	memory=malloc(mem_size);
	if (memory==NULL) perror("allocating memory");

#pragma omp parallel shared(mem_size,memory) private(i) num_threads(num_threads)

/* Do work in each section in parallel */

#pragma omp sections
{
#pragma omp section
{
	for (i=0; i < mem_size/2; i++) {
		memory[i]=0xa5;
	}
}

#pragma omp section
{

	for (i=mem_size/2; i < mem_size; i++) {
		memory[i]=0xa5;
	}
}
}

	/* verify the output */
	for(i=0;i<mem_size;i++) {
		if (memory[i]!=0xa5) {
			printf("Wrong value %x at %d\n",memory[i],i);
		}
	}

	printf("Master thread exiting\n");

}
