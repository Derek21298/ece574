#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include <omp.h>

int main (int argc, char **argv) {

	int num_threads=4;
	int i,tid;

	/* Set number of threads from the command line */
	if (argc>1) {
		num_threads=atoi(argv[1]);
	}


#pragma omp parallel private(i,tid) num_threads(num_threads)
	{

	tid=omp_get_thread_num();

	#pragma omp for schedule(dynamic,1) nowait
	for (i=0; i < 100; i++) {
		if (tid==0) printf("0(%d)\n",i);
		if (tid==1) printf("     1(%d)\n",i);
		if (tid==2) printf("          2(%d)\n",i);
		if (tid==3) printf("               3(%d)\n",i);
		sleep(tid);
	}
}
	printf("Master thread exiting\n");

}
