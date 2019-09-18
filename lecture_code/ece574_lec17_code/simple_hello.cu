#include <stdio.h>

__global__ void kernel(void) {

}

int main(int argc, char **argv) {

	kernel<<<1,1>>>();
	printf("La la\n");
	return 0;
}
