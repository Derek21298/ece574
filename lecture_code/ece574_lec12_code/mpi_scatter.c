/* MPI Scatter Example */
#include <stdio.h>
#include "mpi.h"

#define ARRAYSIZE 1024*1024

int A[ARRAYSIZE];
int B[ARRAYSIZE];


int main(int argc, char **argv) {

	int numtasks, rank;
	int result,i;
	MPI_Status Stat;
	int count;

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

	int sum=0,remote_sum=0;

	/* Calculate our chunk of the sum */
	for(i=0;i<ARRAYSIZE/numtasks;i++) {
		sum+=B[i];
	}

	if (rank==0) {

		/* Master waits for sum from all processes */
		for(i=1;i<numtasks;i++) {
			result = MPI_Recv(&remote_sum, /* buffer */
					1,		/* count */
					MPI_INT,	/* type */
					MPI_ANY_SOURCE,	/* source */
					13,		/* tag */
					MPI_COMM_WORLD,
					&Stat);
			result = MPI_Get_count(&Stat, MPI_INT, &count);
			printf("\tR%d: (%d) Received %d int "
				"from task %d with tag %d \n",
				rank,remote_sum,count,
				Stat.MPI_SOURCE, Stat.MPI_TAG);
			sum+=remote_sum;

		}
		printf("R%d: Total: %d\n",rank,sum);

	}
	else {
		printf("\tR%d Sending back result=%d\n",rank,sum);
		result = MPI_Send(&sum,		/* buffer */
				1,		/* count */
				MPI_INT,	/* type */
				0,		/* destination */
				13,		/* tag */
				MPI_COMM_WORLD);
	}

	MPI_Finalize();
}
