/* See https://gcc.gnu.org/wiki/Offloading */
/* and https://www.xsede.org/documents/234989/378230/XSEDE12_Stampede_Offloading.pdf */

#include <omp.h>
#include <stdlib.h>

int main(int argc, char **argv) {
	const int n = 1024;	/* size */
	float a[n][n];		/* array */
	int i, j;		/* index variables */
	float x;


#pragma offload target(mic)
#pragma omp parallel for shared(a), private(x), schedule(dynamic)
	for(i=0;i<n;i++) {
		for(j=i;j<n;j++) {
			x = (float)(i + j); a[i][j] = x;
		}
	}
}
