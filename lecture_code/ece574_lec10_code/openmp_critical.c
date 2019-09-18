#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

static unsigned char *memory;

int main (int argc, char **argv) {

	int num_threads=4;
	int mem_size=128*1024*1024;	/* 128 MB */
	int i,tid,nthreads;
	int count=0;

	/* Set number of threads from the command line */
	if (argc>1) {
		num_threads=atoi(argv[1]);
	}

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


	#pragma omp for schedule(static) nowait
	for (i=0; i < mem_size; i++) {
		memory[i]=0xa5;
#pragma omp critical(countupdate)
{
		count++;
		if (count%(1<<22)==0) printf("%x\n",count);
}

	}
}
        /* verify the output */
        for(i=0;i<mem_size;i++) {
                if (memory[i]!=0xa5) {
                        printf("Wrong value %x at %d\n",memory[i],i);
                }
        }

	printf("Master thread exiting %d iterations\n",count);

}
