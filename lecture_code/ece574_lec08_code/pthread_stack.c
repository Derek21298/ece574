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

	pthread_attr_t attr;
	size_t stacksize;

	pthread_attr_init(&attr);
	pthread_attr_getstacksize (&attr, &stacksize);

	printf("Default stack size = %li\n", stacksize); /* 8MB on x86? */

	stacksize=16*1024*1024;
	pthread_attr_setstacksize (&attr, stacksize);

	pthread_attr_getstacksize (&attr, &stacksize);

	printf("New stack size = %li\n", stacksize);

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
