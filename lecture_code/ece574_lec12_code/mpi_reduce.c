/* MPI Send Example */
#include <stdio.h>
#include "mpi.h"

#define ARRAYSIZE 1024*1024

int A[ARRAYSIZE];
int B[ARRAYSIZE];


int main(int argc, char **argv) {

	int numtasks, rank;
	int result,i;
//	MPI_Status Stat;
//	int count;

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
		rank, numtasks,rank);

	/* Initialize Array */
	if (rank==0) {
		printf("R0: Initializing array\n");
		for(i=0;i<ARRAYSIZE;i++) {
			A[i]=1;
		}
	}
	MPI_Barrier(MPI_COMM_WORLD);

	MPI_Scatter(A,		/* source buffer */
		ARRAYSIZE/numtasks,	/* count */
		MPI_INT,	/* type */
		B,		/* receive buffer */
		ARRAYSIZE/numtasks,	/* count */
		MPI_INT,	/* type */
		0,		/* root source */
		MPI_COMM_WORLD);

	int sum=0,total_sum=0;

	/* Calculate our chunk of the sum */
	for(i=0;i<ARRAYSIZE/numtasks;i++) {
		sum+=B[i];
	}

	MPI_Reduce(&sum,	/* send data */
		&total_sum,	/* receive data */
		1,		/* count */
		MPI_INT,	/* type */
		MPI_SUM,	/* reduce type */
		0,		/* root */
		MPI_COMM_WORLD);

	if (rank==0) {
		printf("R%d: Total: %d\n",rank,total_sum);
	}

	MPI_Finalize();
}
