#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

struct thread_data_t {
	int thread_number;
};

static int task_num=0;

static pthread_mutexattr_t mattr;
static pthread_mutex_t our_mutex = PTHREAD_MUTEX_INITIALIZER;

#define TOTAL_TASKS 20

void *do_task(void *data) {

	struct thread_data_t *t;
	int mytask,result;
	int finished=0;

	t=(struct thread_data_t *)data;

	/* why can't we do this? */
//	while(task_num<TOTAL_TASKS) {

	while(1) {

		/* blocks; non-blocking can be done by pthread_mutex_trylock */

		result = pthread_mutex_lock(&our_mutex);

		if (task_num>=TOTAL_TASKS) finished=1;
		mytask=task_num;
		task_num++;

		result = pthread_mutex_unlock(&our_mutex);

		/* Why not have this inside lock? */
		if (finished) break;

		printf("\tThread #%d working on task %d\n",
			t->thread_number,
			mytask);
		usleep(250000);

	}
	printf("\tThread #%d exiting\n",t->thread_number);

	pthread_exit(NULL);
}


int main (int argc, char **argv) {

	int num_threads=1;
	pthread_t *threads;
	struct thread_data_t *thread_data;
	int result;
	long int t;

	/* Set number of threads from the command line */
	if (argc>1) {
		num_threads=atoi(argv[1]);
	}

	/* allocate threads */
	threads=calloc(num_threads, sizeof(pthread_t));
	if (threads==NULL) perror("allocating threads");

	/* allocate thread data */
	/* Why must we have unique thread datas? */
	thread_data=calloc(num_threads,sizeof(struct thread_data_t));
	if (thread_data==NULL) perror("allocating thread_data");

	/* Initialize mutex attribute */
	result = pthread_mutexattr_init(&mattr);
	/* set to be shared */
	result = pthread_mutexattr_setpshared(&mattr, PTHREAD_PROCESS_PRIVATE);
	/* Initialize mutex */
	result = pthread_mutex_init(&our_mutex, &mattr);

	printf("Attempting to greedily run %d tasks\n",
		TOTAL_TASKS);

	for(t=0; t<num_threads; t++) {

		thread_data[t].thread_number=t;

		printf("Creating thread %ld\n", t);
		result = pthread_create(
			&threads[t],
			NULL,
			do_task,
			(void *)&thread_data[t]);

		if (result) {
			fprintf(stderr,"ERROR: pthread_create returned %d\n",
					result);
			exit(-1);
		}

	}

	for(t=0;t<num_threads;t++) {
		pthread_join(threads[t],NULL);
	}

	result = pthread_mutexattr_destroy(&mattr);

	result = pthread_mutex_destroy(&our_mutex);


	printf("Master thread exiting\n");

	pthread_exit(NULL);
}
