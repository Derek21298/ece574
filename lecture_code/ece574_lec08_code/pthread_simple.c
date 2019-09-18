#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define NUM_THREADS	5

void *print_message(void *threadid) {

	long int tid;
	tid = (long int)threadid;

	printf("\tIn thread #%ld!\n", tid);

	pthread_exit(NULL);
}

int main (int argc, char **argv) {

	pthread_t threads[NUM_THREADS];
	int result;
	long int t;

	for(t=0; t<NUM_THREADS; t++) {
		printf("Creating thread %ld\n", t);
		result = pthread_create(
			&threads[t],
			NULL,
			print_message,
			(void *)t);

		if (result) {
			fprintf(stderr,"ERROR: pthread_create returned %d\n",
					result);
			exit(-1);
		}
	}

	pthread_exit(NULL);
}
