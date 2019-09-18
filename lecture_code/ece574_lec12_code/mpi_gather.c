/* MPI Gather Example */
#include <stdio.h>
#include "mpi.h"

#define ARRAYSIZE 1024

int A[ARRAYSIZE];
int B[ARRAYSIZE];


int main(int argc, char **argv) {

	int numtasks, rank;
	int result,i;

	/* Initialize MPI */
	result = MPI_Init(&argc,&argv);
	if (result != MPI_SUCCESS) {
		printf ("Error starting MPI program!.\n");
		MPI_Abort(MPI_COMM_WORLD, result);
	}

	/* Get number of tasks and our process number (rank) */
	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);

	printf("R%d: Number of tasks= %d My rank= %d\n",
		rank,numtasks,rank);

	/* Initialize Arrays */
	printf("R%d: Initializing array to %d\n",rank,rank);
	for(i=0;i<ARRAYSIZE;i++) {
		A[i]=rank;
	}

	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Gather(A,		/* send buffer */
		1,		/* count */
		MPI_INT,	/* type */
		B,		/* receive buffer */
		1,		/* count */
		MPI_INT,	/* type */
		0,		/* root source */
		MPI_COMM_WORLD);

	if (rank==0) {

		for(i=0;i<numtasks;i++) {
			printf("R%d: B[%d]=%d\n",rank,i,B[i]);
		}
	}

	MPI_Finalize();
}
