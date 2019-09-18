#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define INIT_VALUE 0xa5

static unsigned char *memory;

struct our_thread_arguments_t {
	int thread_number;
	unsigned char *offset;
	int size;
};


void *init_memory(void *data) {

	struct our_thread_arguments_t *t;
	int i;

	t = (struct our_thread_arguments_t *)data;

	printf("\tIn thread #%d initializing %d bytes at %p\n",
		t->thread_number,
		t->size,
		t->offset);

	for(i=0;i < t->size; i++) {
		t->offset[i]=INIT_VALUE;
	}
	printf("\tFinished %d\n",t->thread_number);

	pthread_exit(NULL);
}


int main (int argc, char **argv) {

	int num_threads=1;
	int mem_size=256*1024*1024;	/* 256 MB */
	pthread_t *threads;
	struct our_thread_arguments_t *thread_data;
	int result;
	long int t;
	int i;

	/* Set number of threads from the command line */
	if (argc>1) {
		num_threads=atoi(argv[1]);
	}

	/* allocate memory */
	memory=malloc(mem_size);
	if (memory==NULL) perror("allocating memory");

	/* allocate threads */
	threads=calloc(num_threads, sizeof(pthread_t));
	if (threads==NULL) perror("allocating threads");

	/* allocate thread data */
	/* Why must we have unique thread datas? */
	thread_data=calloc(num_threads,sizeof(struct our_thread_arguments_t));
	if (thread_data==NULL) perror("allocating thread_data");

	printf("Initializing %d MB of memory using %d threads\n",
		mem_size/(1024*1024),num_threads);

	for(t=0; t<num_threads; t++) {

		thread_data[t].thread_number=t;
		thread_data[t].size=mem_size/num_threads;
		thread_data[t].offset=memory+(t*thread_data->size);

		printf("Creating thread %ld\n", t);
		result = pthread_create(
			&threads[t],
			NULL,
			init_memory,
			(void *)&thread_data[t]);

		if (result) {
			fprintf(stderr,"ERROR: pthread_create returned %d\n",
					result);
			exit(-1);
		}

//		pthread_join(threads[t],NULL);

	}

	for(t=0;t<num_threads;t++) {
		pthread_join(threads[t],NULL);
	}

	printf("Master thread exiting, validating results\n");

	for(i=0;i<mem_size;i++) {
		if (memory[i]!=INIT_VALUE) {
			printf("Uninitialized byte at %d %x\n",i,memory[i]);
		}
	}

	pthread_exit(NULL);
}
