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
void cuda_generic_convolve (int image_size, int xsize, int depth, unsigned char *in, int *matrix, unsigned char *out) {

	int i = blockIdx.x*blockDim.x+threadIdx.x;
	int sum = 0;

	// Calculate the convlution as long as the thread is less than the image size
	if(i < image_size) {

		// Ignore the edges of the image
		if((i >= (xsize*depth+depth)) &&
		   (i <= (image_size-xsize*depth-depth-1)) &&
		   ((i%(xsize*depth)) >= depth) &&
		   ((i%(xsize*depth)) <= (xsize*depth-depth-1))){

			sum = 0;

			// Apply the convolution sum
			sum += matrix[0] * in[i - depth - xsize * depth];
			sum += matrix[1] * in[i - xsize * depth];
			sum += matrix[2] * in[i + depth - xsize * depth];
			sum += matrix[3] * in[i - depth];
			sum += matrix[4] * in[i];
			sum += matrix[5] * in[i + depth];
			sum += matrix[6] * in[i - depth + xsize * depth];
			sum += matrix[7] * in[i + xsize * depth];
			sum += matrix[8] * in[i + depth + xsize * depth];

		}

		// If saturated
		if (sum<0) sum=0;
		if (sum>255) sum=255;

		out[i] = sum;
	}
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

	struct image_t image,new_image;
	long long start_time,load_time,convolve_time;
	long long combine_after=0,combine_before=0;
	long long copy_before=0,copy_after=0,copy2_before=0,copy2_after=0;
	long long store_after,store_before;
	int sobelx[9];
	int sobely[9];

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
	
	// Allocate space for output image
	new_image.x=image.x;
	new_image.y=image.y;
	new_image.depth=image.depth;
	new_image.pixels=(unsigned char *)calloc(image.x*image.y*image.depth,sizeof(char));
	
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
	
	// Allocate space for the sobel matrices
	int *gpu_sobelx;
	cudaMalloc((void **)&gpu_sobelx,9*sizeof(int));

	int *gpu_sobely;
	cudaMalloc((void **)&gpu_sobely,9*sizeof(int));
	
	// Allocate space for the image pixels
	unsigned char *gpu_pixels;
	cudaMalloc((void **)&gpu_pixels, image_size*sizeof(char));

	// Store the filter values to a 1D array
	sobelx[0] = sobel_x_filter[0][0];
	sobelx[1] = sobel_x_filter[0][1];
	sobelx[2] = sobel_x_filter[0][2];
	sobelx[3] = sobel_x_filter[1][0];
	sobelx[4] = sobel_x_filter[1][1];
	sobelx[5] = sobel_x_filter[1][2];
	sobelx[6] = sobel_x_filter[2][0];
	sobelx[7] = sobel_x_filter[2][1];
	sobelx[8] = sobel_x_filter[2][2];

	sobely[0] = sobel_y_filter[0][0];
	sobely[1] = sobel_y_filter[0][1];
	sobely[2] = sobel_y_filter[0][2];
	sobely[3] = sobel_y_filter[1][0];
	sobely[4] = sobel_y_filter[1][1];
	sobely[5] = sobel_y_filter[1][2];
	sobely[6] = sobel_y_filter[2][0];
	sobely[7] = sobel_y_filter[2][1];
	sobely[8] = sobel_y_filter[2][2];

	// Copy the filters and pixels to the GPU
	copy_before = PAPI_get_real_usec();
	cudaMemcpy(gpu_sobelx, sobelx, 9*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_sobely, sobely, 9*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(gpu_pixels, image.pixels, image_size*sizeof(char), cudaMemcpyHostToDevice);
	copy_after = PAPI_get_real_usec();
	printf("Copy host to device: %lld\n", (copy_after-copy_before));

	/* convolution */
	cuda_generic_convolve<<<(image_size+255)/256, 256>>>(image_size, image.x, image.depth, gpu_pixels, gpu_sobelx, gpu_x_pixels);

	cuda_generic_convolve<<<(image_size+255)/256, 256>>>(image_size, image.x, image.depth, gpu_pixels, gpu_sobely, gpu_y_pixels);

	convolve_time=PAPI_get_real_usec();
        printf("Convolve time: %lld\n",convolve_time-load_time);

	// CALL GPU COMBINE CODE
	combine_before = PAPI_get_real_usec();
	cuda_combine<<<(image_size+255)/256, 256>>>(image_size, gpu_x_pixels, gpu_y_pixels, gpu_new_pixels);
	combine_after = PAPI_get_real_usec();
        printf("Combine time: %lld\n",combine_after-combine_before);

	// COPY RESULTS BACK
	copy2_before = PAPI_get_real_usec();
	cudaMemcpy(new_image.pixels, gpu_new_pixels, image_size*sizeof(char), cudaMemcpyDeviceToHost);
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
