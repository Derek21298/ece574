#include <stdio.h>
#include "mpi.h"

int main(int argc, char **argv) {

	int numtasks, rank, len;
	char hostname[MPI_MAX_PROCESSOR_NAME];
	int result;

//	int claimed,provided;
//	MPI_Init_thread(0, 0, MPI_THREAD_MULTIPLE, &provided );
//	MPI_Query_thread( &claimed );
//      printf( "Query thread level= %d  Init_thread level= %d\n", 
//			claimed, provided );


	result = MPI_Init(&argc,&argv);
	if (result != MPI_SUCCESS) {
		printf ("Error starting MPI program!.\n");
		MPI_Abort(MPI_COMM_WORLD, result);
	}

	MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
	MPI_Comm_rank(MPI_COMM_WORLD,&rank);
	MPI_Get_processor_name(hostname, &len);


	if (rank==0) {
		int version,subversion;
		MPI_Get_version (&version,&subversion);
		printf("Version: %d.%d\n",version,subversion);
		printf("Time: %lf\n",MPI_Wtime());
	}

	printf("Number of tasks= %d My rank= %d Running on %s\n",
		numtasks,rank,hostname);

	/*******  do some work *******/

	if (rank==0) {
		printf("Time: %lf\n",MPI_Wtime());

	}

	MPI_Finalize();
}
