#include <stdio.h>
#include <stdlib.h>

int main(int argc, char **argv) {

	int i,j;
	float *x,*y,a;
	long long N=1000*1000*8,loops=1;

	if (argc>1) {
		N=atoll(argv[1]);
	}

	if (argc>2) {
		loops=atoll(argv[2]);
	}

	x=malloc(N*sizeof(float));
	y=malloc(N*sizeof(float));
	if ((x==NULL) || (y==NULL)) {
		printf("Error allocating memory!\n");
		return -1;
	}

	/* Fill the host arrays with values */
	for(i=0;i<N;i++) {
		x[i]=(float)i;
		y[i]=(float)(10.0*i);
	}

	a=5.0;

	/* SAXPY, single-percision y=a*x+y */
	for(j=0;j<loops;j++) {
		for(i=0;i<N;i++) {
			y[i]=a*x[i]+y[i];
		}
	}

	/* results */
	i=100;
	printf("y[%d]=%f y[%lld]=%f\n",i,y[i],N-1,y[N-1]);

	/* 0: a=0, x=0, y=0  ::::::: y=0 */
        /* 1: a=1, x=1, y=10 ::::::: y=11 */
        /* 2: a=2, x=2, y=20 ::::::: y=24 */
        /* 3: a=3, x=3, y=30 ::::::: y=39 */
        /* 4: a=4, x=4, y=40 ::::::: y=56 */
	/* ... */
	/* 100: a=100, x=100, y=1000 y=11000 */

	free(x);
	free(y);

	return 0;
}

