#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

int main (int argc, char **argv) {

	int nthreads,tid;

/* Fork a parallel region, each thread having private copy of tid */
#pragma omp parallel private(tid)
	{
	tid=omp_get_thread_num();
	printf("\tInside of thread %d\n",tid);

	if (tid==0) {
		nthreads=omp_get_num_threads();
		printf("This is the master thread, there are %d threads\n",
			nthreads);
	}

	}

	/* End of block, waits and joins automatically */


	return 0;
}
