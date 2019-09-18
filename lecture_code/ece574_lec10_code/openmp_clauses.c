#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

int main (int argc, char **argv) {

	int nthreads,tid,blah=5;
	int run_parallel=1;

	if (argc>1) {
		run_parallel=atoi(argv[1]);
	}

/* Fork a parallel region, each thread having private copy of tid */
#pragma omp parallel if (run_parallel) private(tid) shared(nthreads) \
	default(shared) firstprivate(blah)
	{
	tid=omp_get_thread_num();
	printf("\tInside of thread %d blah=%d\n",tid,blah);

	if (tid==0) {
		nthreads=omp_get_num_threads();
		printf("This is the master thread, there are %d threads\n",
			nthreads);
	}

	}

	/* End of block, waits and joins automatically */


	return 0;
}
