/* Example sobel code for ECE574 -- Spring 2019 */
/* By Vince Weaver <vincent.weaver@maine.edu> */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>

#include <jpeglib.h>

#include <cuda.h>

#include <papi.h>

/* Filters */
static int sobel_x_filter[3][3]={{-1,0,+1},{-2,0,+2},{-1,0,+1}};
static int sobel_y_filter[3][3]={{-1,-2,-1},{0,0,0},{1,2,+1}};

/* Structure describing the image */
struct image_t {
	int x;
	int y;
	int depth;	/* bytes */
	unsigned char *pixels;
};

struct convolve_data_t {
	struct image_t *old;
	struct image_t *newt;
	int (*filter)[3][3];
	int ystart;
	int yend;
};

__global__
void cuda_generic_convolve (int n, char *in, int *matrix, char *out) {
}	

__global__
void cuda_combine (int n, unsigned char *in_x, 
		unsigned char *in_y, unsigned char *out) {

	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int result;

	if(i<n) {
 
		// Do the combine
		// Cast to double for GPU
		result=sqrt(
			(double)(in_x[i]*in_x[i])+
			(double)(in_y[i]*in_y[i])
			);
		
		// If saturated
		if (result>255) result=255;
		if (result<0) result=0;
		out[i]=result;
	}
}

/* very inefficient convolve code */
static void *generic_convolve(void *argument) {

	int x,y,k,l,d;
	uint32_t color;
	int sum,depth,width;

	struct image_t *old;
	struct image_t *newt;
	int (*filter)[3][3];
	struct convolve_data_t *data;
	int ystart, yend;

	/* Convert from void pointer to the actual data type */
	data=(struct convolve_data_t *)argument;
	old=data->old;
	newt=data->newt;
	filter=data->filter;

	ystart=data->ystart;
	yend=data->yend;

	depth=old->depth;
	width=old->x*old->depth;

	if (ystart==0) ystart=1;
	if (yend==old->y) yend=old->y-1;

	for(d=0;d<3;d++) {
	   for(x=1;x<old->x-1;x++) {
	     for(y=ystart;y<yend;y++) {
		sum=0;
		for(k=-1;k<2;k++) {
		   for(l=-1;l<2;l++) {
			color=old->pixels[((y+l)*width)+(x*depth+d+k*depth)];
			sum+=color * (*filter)[k+1][l+1];
		   }
		}

		if (sum<0) sum=0;
		if (sum>255) sum=255;

		newt->pixels[(y*width)+x*depth+d]=sum;
	     }
	   }
	}

	return NULL;
}

static int combine(struct image_t *s_x,
			struct image_t *s_y,
			struct image_t *newt) {
	
	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int out;

	for(i=0;i<( s_x->depth * s_x->x * s_x->y );i++) {
 
		out=sqrt(
			(s_x->pixels[i]*s_x->pixels[i])+
			(s_y->pixels[i]*s_y->pixels[i])
			);
		if (out>255) out=255;
		if (out<0) out=0;
		newt->pixels[i]=out;
	}

	return 0;
} 

static int load_jpeg(char *filename, struct image_t *image) {

	FILE *fff;
	struct jpeg_decompress_struct cinfo;
	struct jpeg_error_mgr jerr;
	JSAMPROW output_data;
	unsigned int scanline_len;
	int scanline_count=0;

	fff=fopen(filename,"rb");
	if (fff==NULL) {
		fprintf(stderr, "Could not load %s: %s\n",
			filename, strerror(errno));
		return -1;
	}

	/* set up jpeg error routines */
	cinfo.err = jpeg_std_error(&jerr);

	/* Initialize cinfo */
	jpeg_create_decompress(&cinfo);

	/* Set input file */
	jpeg_stdio_src(&cinfo, fff);

	/* read header */
	jpeg_read_header(&cinfo, TRUE);

	/* Start decompressor */
	jpeg_start_decompress(&cinfo);

	printf("output_width=%d, output_height=%d, output_components=%d\n",
		cinfo.output_width,
		cinfo.output_height,
		cinfo.output_components);

	image->x=cinfo.output_width;
	image->y=cinfo.output_height;
	image->depth=cinfo.output_components;

	scanline_len = cinfo.output_width * cinfo.output_components;
	image->pixels=(unsigned char *)malloc(cinfo.output_width * cinfo.output_height * cinfo.output_components);

	while (scanline_count < cinfo.output_height) {
		output_data = (image->pixels + (scanline_count * scanline_len));
		jpeg_read_scanlines(&cinfo, &output_data, 1);
		scanline_count++;
	}

	/* Finish decompressing */
	jpeg_finish_decompress(&cinfo);

	jpeg_destroy_decompress(&cinfo);

	fclose(fff);

	return 0;
}

static int store_jpeg(const char *filename, struct image_t *image) {

	struct jpeg_compress_struct cinfo;
	struct jpeg_error_mgr jerr;
	int quality=90; /* % */
	int i;

	FILE *fff;

	JSAMPROW row_pointer[1];
	int row_stride;

	/* setup error handler */
	cinfo.err = jpeg_std_error(&jerr);

	/* initialize jpeg compression object */
	jpeg_create_compress(&cinfo);

	/* Open file */
	fff = fopen(filename, "wb");
	if (fff==NULL) {
		fprintf(stderr, "can't open %s: %s\n",
			filename,strerror(errno));
		return -1;
	}

	jpeg_stdio_dest(&cinfo, fff);

	/* Set compression parameters */
	cinfo.image_width = image->x;
	cinfo.image_height = image->y;
	cinfo.input_components = image->depth;
	cinfo.in_color_space = JCS_RGB;
	jpeg_set_defaults(&cinfo);
	jpeg_set_quality(&cinfo, quality, TRUE);

	/* start compressing */
	jpeg_start_compress(&cinfo, TRUE);

	row_stride=image->x*image->depth;

	for(i=0;i<image->y;i++) {
		row_pointer[0] = & image->pixels[i * row_stride];
		jpeg_write_scanlines(&cinfo, row_pointer, 1);
	}

	/* finish compressing */
	jpeg_finish_compress(&cinfo);

	/* close file */
	fclose(fff);

	/* clean up */
	jpeg_destroy_compress(&cinfo);

	return 0;
}

int main(int argc, char **argv) {

	struct image_t image,sobel_x,sobel_y,new_image;
	struct convolve_data_t sobel_data[2];
	long long start_time,load_time,convolve_time;
	long long combine_after=0,combine_before=0;
	long long copy_before=0,copy_after=0,copy2_before=0,copy2_after=0;
	long long store_after,store_before;

	/* Check command line usage */
	if (argc<2) {
		fprintf(stderr,"Usage: %s image_file\n",argv[0]);
		return -1;
	}

	PAPI_library_init(PAPI_VER_CURRENT);

	start_time=PAPI_get_real_usec();

	/* Load an image */
	load_jpeg(argv[1],&image);

	load_time=PAPI_get_real_usec();
	printf("Load time: %lld\n",load_time-start_time);

	int image_size = image.x*image.y*image.depth;
	
	/* Allocate space for output image */
	new_image.x=image.x;
	new_image.y=image.y;
	new_image.depth=image.depth;
	new_image.pixels=(unsigned char *)calloc(image.x*image.y*image.depth,sizeof(char));

	/* Allocate space for output image */
	sobel_x.x=image.x;
	sobel_x.y=image.y;
	sobel_x.depth=image.depth;
	sobel_x.pixels=(unsigned char *)calloc(image.x*image.y*image.depth,sizeof(char));

	/* Allocate space for output image */
	sobel_y.x=image.x;
	sobel_y.y=image.y;
	sobel_y.depth=image.depth;
	sobel_y.pixels=(unsigned char *)calloc(image.x*image.y*image.depth,sizeof(char));

	// START ALLOCATING SPACE ON GPU
	/* Allocate space for gpu image */
	unsigned char *gpu_new_pixels;
	cudaMalloc((void **)&gpu_new_pixels,image_size*sizeof(char));

	/* Allocate space for gpu image */
	unsigned char *gpu_x_pixels;
	cudaMalloc((void **)&gpu_x_pixels,image_size*sizeof(char));

	/* Allocate space for gpu image */
	unsigned char *gpu_y_pixels;
	cudaMalloc((void **)&gpu_y_pixels,image_size*sizeof(char));
	
	/* convolution */
	sobel_data[0].old=&image;
	sobel_data[0].newt=&sobel_x;
	sobel_data[0].filter=&sobel_x_filter;
	sobel_data[0].ystart=0;
	sobel_data[0].yend=image.y;
	generic_convolve((void *)&sobel_data[0]);

	sobel_data[1].old=&image;
	sobel_data[1].newt=&sobel_y;
	sobel_data[1].filter=&sobel_y_filter;
	sobel_data[1].ystart=0;
	sobel_data[1].yend=image.y;
	generic_convolve((void *)&sobel_data[1]);

	convolve_time=PAPI_get_real_usec();
        printf("Convolve time: %lld\n",convolve_time-load_time);

	// COPY THE PIXELS TO THE GPU WITH cudaMemcpy()
	copy_before = PAPI_get_real_usec();
	cudaMemcpy(gpu_x_pixels, sobel_x.pixels, image_size, cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_y_pixels, sobel_y.pixels, image_size, cudaMemcpyHostToDevice);
	copy_after = PAPI_get_real_usec();
	printf("Copy host to device: %lld\n",(copy_after-copy_before));

	// CALL GPU COMBINE CODE
	combine_before = PAPI_get_real_usec();
	cuda_combine<<<(image_size+255)/256, 256>>>(image_size, gpu_x_pixels, gpu_y_pixels, gpu_new_pixels);
	combine_after = PAPI_get_real_usec();
        printf("Combine time: %lld\n",combine_after-combine_before);

	printf("Image size: %d\n", image_size);

	// COPY RESULTS BACK
	copy2_before = PAPI_get_real_usec();
	cudaMemcpy(new_image.pixels, gpu_new_pixels, image_size, cudaMemcpyDeviceToHost);
	copy2_after = PAPI_get_real_usec();
	printf("Copy device to host: %lld\n",(copy2_after-copy2_before));

	store_before=PAPI_get_real_usec();

	/* Write data back out to disk */
	store_jpeg("out.jpg",&new_image);

	store_after=PAPI_get_real_usec();
        printf("Store time: %lld\n",store_after-store_before);

	/* Print timing results */
	printf("Total time = %lld\n",store_after-start_time);

	return 0;
}
