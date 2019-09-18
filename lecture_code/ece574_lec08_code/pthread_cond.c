#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define NUM_THREADS  2

#define TOTAL_COUNT 20
#define COUNT_LIMIT 10

static int count = 0;
pthread_mutex_t count_mutex;
pthread_cond_t count_threshold_cv;

static void *inc_count(void *t) {

	int i;
//	long my_id = (long)t;

	for (i=0; i<TOTAL_COUNT; i++) {
		pthread_mutex_lock(&count_mutex);
		count++;

		if (count == COUNT_LIMIT) {
			pthread_cond_signal(&count_threshold_cv);
			printf("Reached limit, sending signal\n");
		}

		printf("Unlocking mutex\n");

		pthread_mutex_unlock(&count_mutex);

		/* Do some "work" */
		usleep(10);
	}

	pthread_exit(NULL);
}

void *watch_count(void *t) {

	long my_id = (long)t;

	printf("Starting watch_count(): thread %ld\n", my_id);

	pthread_mutex_lock(&count_mutex);
	while (count<COUNT_LIMIT) {
		/* This unlocks the mutex while it waits */
		pthread_cond_wait(&count_threshold_cv, &count_mutex);
     		printf("watch_count(): thread %ld Condition signal received.\n", my_id);
	}
	pthread_mutex_unlock(&count_mutex);
	pthread_exit(NULL);
}


int main (int argc, char **argv) {

	int i;
	long t1=1, t2=2;
	pthread_t threads[NUM_THREADS];

	/* Initialize mutex */
	pthread_mutex_init(&count_mutex, NULL);
	/* Initialize condition variable */
	pthread_cond_init (&count_threshold_cv, NULL);

	pthread_create(&threads[0], NULL, watch_count, (void *)t1);
	pthread_create(&threads[1], NULL, inc_count, (void *)t2);

	/* Wait for threads to complete */
	for (i=0; i<NUM_THREADS; i++) {
		pthread_join(threads[i], NULL);
	}

	/* Cleanup */
	pthread_mutex_destroy(&count_mutex);
	pthread_cond_destroy(&count_threshold_cv);

	pthread_exit(NULL);

}
