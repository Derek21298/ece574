#include <stdio.h>

__global__ void add (int a, int b, int *c) {
	*c=a+b;
}

int main(int argc, char **argv) {

	int c;
	int *dev_c;

	/* Allocate memory on device */
	/* Note, the pointer returned is *not* valid on the host */
	/* and dereferencing it will not work */
	cudaMalloc( (void **)&dev_c,sizeof(int));

	add<<<1,1>>>(3,4,dev_c);

	cudaMemcpy(	&c,
			dev_c,
			sizeof(int),
			cudaMemcpyDeviceToHost);

	printf("3+4=%d\n",c);

	cudaFree(dev_c);

	return 0;

}
