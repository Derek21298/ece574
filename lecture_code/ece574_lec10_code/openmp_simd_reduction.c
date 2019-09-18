#include <stdio.h>
#include <stdlib.h>

#include <omp.h>

#define N 10000000

long long a[N];

int main (int argc, char **argv) {

	int num_threads=4;
	int i,j;

	long long sum=0;

	if (argc>1) {
		num_threads=atoi(argv[1]);
	}

	for(j=0;j<N;j++) {
		a[j]=j;
	}

/* Fork a parallel region, each thread having private copy of tid */
#pragma omp parallel for simd reduction(+:sum) schedule(static,8) num_threads(num_threads)
	for(i = 0; i < N; i++) {
		/* Can all threads run at once? */
		sum = sum + i*a[i];
	}


	printf("sum=%lld\n",sum);

	return 0;
}
