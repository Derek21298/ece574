/* Based on code from here: http://devblogs.nvidia.com/parallelforall/easy-introduction-cuda-c-and-c/ */

#include <stdio.h>
#include <stdlib.h>

/* Calculate SAXPY, single-precision vector math 	*/
/* y[i]=a*x[i]+y[i]					*/

__global__
void saxpy (int n, float a, float *x, float *y) {

	int i=blockIdx.x*blockDim.x+threadIdx.x;

	/* Only run calculation if we are in range */
	/* where i is valid.  It can be out of range */
	/* if our vector is shorter than a */
	/* multiple of the blocksize */

	if (i<n) {
		y[i]=a*x[i]+y[i];
	}
}

int main(int argc, char **argv) {

	int i,j;
	float *x, *y, *dev_x, *dev_y;
	float a;
	long long N=(1000*1000*8),loops=1;

	if (argc>1) {
		N=atoll(argv[1]);
	}

	if (argc>2) {
		loops=atoll(argv[2]);
	}

	/* Allocate vectors on CPU */
	x=(float *)malloc(N*sizeof(float));
	y=(float *)malloc(N*sizeof(float));

	/* Allocate vectors on GPU */
	cudaMalloc((void **)&dev_x,N*sizeof(float));
	cudaMalloc((void **)&dev_y,N*sizeof(float));

	/* Initialize the host vectors */
	for(i=0;i<N;i++) {
		x[i]=(float)i;
		y[i]=(float)(10.0*i);
	}

	cudaMemcpy(dev_x,x,N*sizeof(float),cudaMemcpyHostToDevice);
	cudaMemcpy(dev_y,y,N*sizeof(float),cudaMemcpyHostToDevice);

	printf("Size: %d\n",(N+255)/256);

	a=5.0;

	for(j=0;j<loops;j++) {
		/* Perform SAXPY */
		saxpy<<<(N+255)/256,256>>>(N,a,dev_x,dev_y);
	}	

	// make the host block until the device is finished
	cudaDeviceSynchronize();

	// check for error
	cudaError_t error = cudaGetLastError();
	if (error != cudaSuccess) {
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
 	}
	
	cudaMemcpy(y,dev_y,N*sizeof(float),cudaMemcpyDeviceToHost);

	/* results */
	i=100;
	printf("y[%d]=%f, y[%lld]=%f\n",i,y[i],N-1,y[N-1]);

	/* y[i]=a*x[i]+y[i] */
	/* 0: a=5, x=0, y=0  ::::::: y=0 */
	/* 1: a=5, x=1, y=10 ::::::: y=15 */
	/* 2: a=5, x=2, y=20 ::::::: y=30 */
	/* 3: a=5, x=3, y=30 ::::::: y=45 */
	/* 4: a=5, x=4, y=40 ::::::: y=60 */
	/* ... */
	/* 100: a=5, x=100, y=1000 y=1500 */

	cudaFree(dev_x);
	cudaFree(dev_y);

	return 0;
}

