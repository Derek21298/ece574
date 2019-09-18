/* MPI Send Example */
#include <stdio.h>
#include <unistd.h>
#include "mpi.h"


int main(int argc, char **argv) {

	int numtasks, rank;
	int result;

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

	printf("R%d sleeping %d seconds\n",rank,rank*2);
	sleep(2*rank);

	printf("*** R%d waiting at barrier\n",rank);
	MPI_Barrier(MPI_COMM_WORLD);

	printf("R%d continuing\n",rank);
	sleep(1);

	MPI_Finalize();
}
