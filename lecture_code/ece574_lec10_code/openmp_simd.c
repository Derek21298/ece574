#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

static unsigned char *memory;

int main (int argc, char **argv) {

	int num_threads=1;
	int mem_size=256*1024*1024;	/* 256 MB */
	int i,tid,nthreads;

	/* Set number of threads from the command line */
	if (argc>1) {
		num_threads=atoi(argv[1]);
	}

//#pragma omp declare simd
//#pragma omp simd reduction(+:x) 
//#pragma ivdep

	/* allocate memory */
	memory=malloc(mem_size);
	if (memory==NULL) perror("allocating memory");

#pragma omp parallel shared(mem_size,memory) private(i,tid) num_threads(num_threads)
{

	tid=omp_get_thread_num();
	if (tid==0) {
		nthreads=omp_get_num_threads();
		printf("Initializing %d MB of memory using %d threads\n",
			mem_size/(1024*1024),nthreads);
	}


	#pragma omp for simd
	for (i=0; i < mem_size; i++) {
		memory[i]=0xa5;
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
