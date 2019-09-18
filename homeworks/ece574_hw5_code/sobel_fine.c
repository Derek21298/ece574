#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>

#include <jpeglib.h>

#include "papi.h"

#define OMP_NUM_THREADS 32

static int sobel_x_filter[3][3]={{-1,0,+1},{-2,0,+2},{-1,0,+1}};
static int sobel_y_filter[3][3]={{-1,-2,-1},{0,0,0},{1,2,+1}};

struct image_t {
	int x;
	int y;
	int depth;	/* bytes */
	unsigned char *pixels;
};


static void generic_convolve(struct image_t *input_image,
				struct image_t *output_image,
				int filter[3][3]) {


	/* Look at the above image_t definition			*/
	/* You can use input_image->x (width)  			*/
	/*             input_image->y (height) 			*/
	/*         and input_image->depth (bytes per pixel)	*/
	/* input_image->pixels points to the RGB values		*/

	/******************/
	/* Your code here */


	// Create variables to conduct indexing
	int x, y, color;
	int sum;
	int xsize = input_image->x;
	int ysize = input_image->y;

	// Index through all pixels and their colors in an image
	// Start 1 pixel in from the corner for ease

// Tell OMP im making this section parallel and that 
// my iterator variables are private, and the ysize of
// the image is shared so the image is broken up
#pragma omp parallel shared(ysize) private(x,y,sum,color) num_threads(OMP_NUM_THREADS)
{

	// Make the scheduler static
	#pragma omp for	schedule(static) nowait
	for(x = 1; x < xsize - 1; x++) {
		for(y = 1; y < ysize - 1; y++) {
			for(color = 0; color < 3; color++) {
				
				sum = 0;		

				// Apply the filter to each individual pixel
				sum += filter[0][0] * input_image->pixels[((y-1)*xsize*3)+((x-1)*3)+(color)];
				sum += filter[1][0] * input_image->pixels[((y-1)*xsize*3)+(x*3)+(color)];
				sum += filter[2][0] * input_image->pixels[((y-1)*xsize*3)+((x+1)*3)+(color)];
				sum += filter[0][1] * input_image->pixels[(y*xsize*3)+((x-1)*3)+(color)];
				sum += filter[1][1] * input_image->pixels[(y*xsize*3)+(x*3)+(color)];
				sum += filter[2][1] * input_image->pixels[(y*xsize*3)+((x+1)*3)+(color)];
				sum += filter[0][2] * input_image->pixels[((y+1)*xsize*3)+((x-1)*3)+(color)];
				sum += filter[1][2] * input_image->pixels[((y+1)*xsize*3)+(x*3)+(color)];
				sum += filter[2][2] * input_image->pixels[((y+1)*xsize*3)+((x+1)*3)+(color)];
	
				// If the sum is over 255, saturate the pixel accordingly
				if(sum > 255) sum = 255;
				if(sum < 0) sum = 0;

				// Set the convolved pixel to the output image
				output_image->pixels[(y*xsize*3)+(x*3)+(color)] = sum;
			}
		}
	}
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
	image->pixels=malloc(cinfo.output_width * cinfo.output_height * cinfo.output_components);

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

static int store_jpeg(char *filename, struct image_t *image) {

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

static int combine(struct image_t *sobel_x, struct image_t *sobel_y,
		struct image_t *new_image) {

	/******************/
	/* your code here */

	// Initialize width, height, and for loop variables
	int x, y, color;
	int xsize = new_image->x;
	int ysize = new_image->y;
	int result;

#pragma omp parallel shared(ysize) private(x,y,color) num_threads(OMP_NUM_THREADS)
{

	#pragma omp for
	// Go through each pixel (x,y) and the RGB colors
	for(x = 1; x < xsize-1; x++) {
		for(y = 1; y < ysize-1; y++) {
			for(color = 0; color < 3; color++) {
					
				// Result = SQRT(sobelX^2 + sobelY^2) -> combine the results of the sobels
				result = sqrt((sobel_x->pixels[(y*xsize*3)+(x*3)+(color)] * sobel_x->pixels[(y*xsize*3)+(x*3)+(color)]) + (sobel_y->pixels[(y*xsize*3)+(x*3)+(color)] * sobel_y->pixels[(y*xsize*3)+(x*3)+(color)]));

				// Need to saturate your results if larger than 255 or less than 0
				if(result > 255) result = 255;
				if(result < 0) result = 0;

				// Set the output pixel to the combination of sobel x and sobel y
				new_image->pixels[(y*xsize*3)+(x*3)+(color)] = result;
			}
		}
	}		
}
	/******************/

	return 0;
}

int main(int argc, char **argv) {

	struct image_t image,sobel_x,sobel_y,new_image;
	long long start, stop, result;
	long long begin, end;

	/* Check command line usage */
	if (argc<2) {
		fprintf(stderr,"Usage: %s image_file\n",argv[0]);
		return -1;
	}
	begin = PAPI_get_real_usec();
	// Get the time it takes to load the jpeg
	start = PAPI_get_real_usec();
	/* Load an image */
	load_jpeg(argv[1],&image);
	stop = PAPI_get_real_usec();
	result = stop - start;
	printf("JPEG Load Time: %lld us\n", result);

	/* Allocate space for output image */
	new_image.x=image.x;
	new_image.y=image.y;
	new_image.depth=image.depth;
	new_image.pixels=malloc(image.x*image.y*image.depth*sizeof(char));

	/* Allocate space for output image */
	sobel_x.x=image.x;
	sobel_x.y=image.y;
	sobel_x.depth=image.depth;
	sobel_x.pixels=malloc(image.x*image.y*image.depth*sizeof(char));

	/* Allocate space for output image */
	sobel_y.x=image.x;
	sobel_y.y=image.y;
	sobel_y.depth=image.depth;
	sobel_y.pixels=malloc(image.x*image.y*image.depth*sizeof(char));

	// Get the time it takes to convolve
	start = PAPI_get_real_usec();
	/* convolution */
	generic_convolve(&image,&sobel_x, sobel_x_filter);

	generic_convolve(&image,&sobel_y, sobel_y_filter);
	stop = PAPI_get_real_usec();
	result = stop - start;
	printf("Convolve Time: %lld us\n", result);

	// Get the time it takes to combine
	start = PAPI_get_real_usec();
	combine(&sobel_x,&sobel_y,&new_image);
	stop = PAPI_get_real_usec();
	result = stop - start;
	printf("Combine Time: %lld us\n", result);

	// Get the time it takes to store the jpeg
	start = PAPI_get_real_usec();
	store_jpeg("out.jpg",&new_image);
	stop = PAPI_get_real_usec();
	result = stop - start;
	printf("JPEG Store Time: %lld us\n", result);
	end = PAPI_get_real_usec();
	printf("Total Time: %lld us\n", end-begin);

	PAPI_shutdown();

	return 0;
}
