/* Example openCL sobel code for ECE574 -- Spring 2019 */
/* By Vince Weaver <vincent.weaver@maine.edu> */

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <errno.h>
#include <math.h>

/* We are using some OpenCL 1.2 interfaces that have been deprecated	*/
/* These defines will silence the warnings for now.			*/
#define CL_TARGET_OPENCL_VERSION 220
#define CL_USE_DEPRECATED_OPENCL_1_2_APIS

#include <CL/cl.h>

#include <jpeglib.h>

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

/* YOUR CODE HERE: Update the kernels */


// Creates a string of code for the combine kernel
// Combine square of the sobel_x and sobel_y
// Take square root 
// If the result is saturated, make output 255
const char *KernelSource_combine = "\n" \
"__kernel void opencl_combine(			\n" \
"   const unsigned int n,			\n" \
"   __global unsigned char* in_x,		\n" \
"   __global unsigned char* in_y,		\n" \
"   __global unsigned char* out) {		\n" \
"						\n" \
"   int i = get_global_id(0);			\n" \
"   int result;                                 \n" \
"                                               \n" \
"    if(i<n) {                                  \n" \
"	result = sqrt(                          \n" \
"		(float)(in_x[i]*in_x[i])+       \n" \
"		(float)(in_y[i]*in_y[i])        \n" \
"		);                              \n" \
"                                               \n" \
"	if(result > 255) result = 255;          \n" \
"	if(result < 0) result = 0;              \n" \
"			                        \n" \
"   out[i]=result;				\n" \
" }						\n" \
"}						\n" \
"\n";

// Creates a string of code for the convolve kernel
// Ignore the outer boundary of the image
// Apply the filter to each pixel
const char *KernelSource_convolve = "\n" \
"__kernel void opencl_convolve(					\n" \
"   const unsigned int pitch,					\n" \
"   const unsigned int size,					\n" \
"   __global unsigned char* in,					\n" \
"   __global int* matrix,					\n" \
"   __global unsigned char* out){				\n" \
"   int i = get_global_id(0);					\n" \
"   int sum = 0;						\n" \
"								\n" \
"   if(i < size) {						\n" \
"	    							\n" \
"	    if((i >= (pitch+3)) &&				\n" \
"	       (i <= (size-pitch-3-1)) &&			\n" \
"	       ((i%(pitch)) >= 3) &&				\n" \
"	       ((i%(pitch)) <= (pitch-3-1))){			\n" \		
"		    						\n" \
"		    sum = 0;					\n" \
"								\n" \
"		    sum += matrix[0] * in[i-3-pitch];	    	\n" \
"	    	    sum += matrix[1] * in[i-pitch];		\n" \
"		    sum += matrix[2] * in[i+3-pitch];		\n" \
"		    sum += matrix[3] * in[i-3];			\n" \
"		    sum += matrix[4] * in[i];			\n" \
"		    sum += matrix[5] * in[i+3];			\n" \
"		    sum += matrix[6] * in[i-3+pitch];		\n" \
"		    sum += matrix[7] * in[i+pitch];		\n" \
"		    sum += matrix[8] * in[i+3+pitch];		\n" \
"								\n" \
"	    }							\n" \
"  								\n" \
"	    if(sum < 0) sum = 0;				\n" \
"	    if(sum > 255) sum = 255;				\n" \
"								\n" \
"   	    out[i]=sum;						\n" \
"   }								\n" \
"}                                                              \n" \
"\n";


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

/* Note: OpenCL does not include a strerror() function */

static const char *cl_getErrorString(cl_int error) {
	switch(error){
		/* run-time and JIT compiler errors */
		case CL_SUCCESS:
			return "CL_SUCCESS";				// 0
		case CL_DEVICE_NOT_FOUND:
			return "CL_DEVICE_NOT_FOUND";			// -1
		case CL_DEVICE_NOT_AVAILABLE:
			return "CL_DEVICE_NOT_AVAILABLE";		// -2
		case CL_COMPILER_NOT_AVAILABLE:
			return "CL_COMPILER_NOT_AVAILABLE";		// -3
		case CL_MEM_OBJECT_ALLOCATION_FAILURE:
			return "CL_MEM_OBJECT_ALLOCATION_FAILURE";	// -4
		case CL_OUT_OF_RESOURCES:
			return "CL_OUT_OF_RESOURCES";			// -5
		case -6:
			return "CL_OUT_OF_HOST_MEMORY";			// -6
		case -7:
			return "CL_PROFILING_INFO_NOT_AVAILABLE";	// -7
		case -8:
			return "CL_MEM_COPY_OVERLAP";			// -8
		case -9:
			return "CL_IMAGE_FORMAT_MISMATCH";		// -9
		case -10:
			return "CL_IMAGE_FORMAT_NOT_SUPPORTED";		// -10
		case -11:
			return "CL_BUILD_PROGRAM_FAILURE";		// -11
		case -12:
			return "CL_MAP_FAILURE";			// -12
		case -13:
			return "CL_MISALIGNED_SUB_BUFFER_OFFSET";	// -13
		case -14:
			return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";	// -14
		case -15:
			return "CL_COMPILE_PROGRAM_FAILURE";		// -15
		case -16:
			return "CL_LINKER_NOT_AVAILABLE";		// -16
		case -17:
			return "CL_LINK_PROGRAM_FAILURE";		// -17
		case -18:
			return "CL_DEVICE_PARTITION_FAILED";		// -18
		case -19:
			return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";	// -19

		/* compile-time errors */
		case -30:
			return "CL_INVALID_VALUE";
		case -31:
			return "CL_INVALID_DEVICE_TYPE";
		case -32:
			return "CL_INVALID_PLATFORM";
		case -33:
			return "CL_INVALID_DEVICE";
		case -34:
			return "CL_INVALID_CONTEXT";
		case -35:
			return "CL_INVALID_QUEUE_PROPERTIES";
		case -36:
			return "CL_INVALID_COMMAND_QUEUE";
		case -37:
			return "CL_INVALID_HOST_PTR";
		case -38:
			return "CL_INVALID_MEM_OBJECT";
		case -39:
			return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40:
			return "CL_INVALID_IMAGE_SIZE";
		case -41:
			return "CL_INVALID_SAMPLER";
		case -42:
			return "CL_INVALID_BINARY";
		case -43:
			return "CL_INVALID_BUILD_OPTIONS";
		case -44:
			return "CL_INVALID_PROGRAM";
		case -45:
			return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46:
			return "CL_INVALID_KERNEL_NAME";
		case -47:
			return "CL_INVALID_KERNEL_DEFINITION";
		case -48:
			return "CL_INVALID_KERNEL";
		case -49:
			return "CL_INVALID_ARG_INDEX";
		case -50:
			return "CL_INVALID_ARG_VALUE";
		case -51:
			return "CL_INVALID_ARG_SIZE";
		case -52:
			return "CL_INVALID_KERNEL_ARGS";
		case -53:
			return "CL_INVALID_WORK_DIMENSION";
		case -54:
			return "CL_INVALID_WORK_GROUP_SIZE";
		case -55:
			return "CL_INVALID_WORK_ITEM_SIZE";
		case -56:
			return "CL_INVALID_GLOBAL_OFFSET";
		case -57:
			return "CL_INVALID_EVENT_WAIT_LIST";
		case -58:
			return "CL_INVALID_EVENT";
		case -59:
			return "CL_INVALID_OPERATION";
		case -60:
			return "CL_INVALID_GL_OBJECT";
		case -61:
			return "CL_INVALID_BUFFER_SIZE";
		case -62:
			return "CL_INVALID_MIP_LEVEL";
		case -63:
			return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64:
			return "CL_INVALID_PROPERTY";
		case -65:
			return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66:
			return "CL_INVALID_COMPILER_OPTIONS";
		case -67:
			return "CL_INVALID_LINKER_OPTIONS";
		case -68:
			return "CL_INVALID_DEVICE_PARTITION_COUNT";

		/* extension errors */
		case -1000:
			return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";// -1000
		case -1001:
			return "CL_PLATFORM_NOT_FOUND_KHR";		// -1001
		case -1002:
			return "CL_INVALID_D3D10_DEVICE_KHR";		// -1002
		case -1003:
			return "CL_INVALID_D3D10_RESOURCE_KHR";		// -1003
		case -1004:
			return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";// -1004
		case -1005:
			return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";	// -1005
		default:
			return "Unknown OpenCL error";
	}
}



int main(int argc, char **argv) {

	struct image_t image,sobel_x,sobel_y,new_image;

	long long start_time,load_before,load_after;
	long long convolve_after,convolve_before;
	long long combine_after,combine_before;
	long long copy_before,copy_after,copy2_before,copy2_after;
	long long store_after,store_before;


	/* OpenCL defines */
#define MAX_DEVICES	4
#define MAX_PLATFORMS	5
	cl_int err;
	cl_device_id device_id;
	cl_platform_id platform[MAX_PLATFORMS];
	cl_uint num_platforms;
	size_t returned_size = 0;
	cl_char platform_name[MAX_PLATFORMS][1024], platform_prof[MAX_PLATFORMS][1024],
		platform_vers[MAX_PLATFORMS][1024], platform_exts[MAX_PLATFORMS][1024];
	cl_uint num_devices;
	cl_device_id devices[MAX_DEVICES];
	cl_context context;
	cl_command_queue commands;
	cl_program program_combine,program_convolve;
	cl_kernel kernel_combine,kernel_convolve;
	size_t local;
	size_t global;
	int i;
	int which_platform=0;

	/****************************/
	/* Check command line usage */
	/****************************/
	if (argc<2) {
		fprintf(stderr,"Usage: %s image_file\n",argv[0]);
		return -1;
	}
	if (argc>2) {
		which_platform=atoi(argv[2]);
	}

	/***************/
	/* Set up PAPI */
	/***************/
	PAPI_library_init(PAPI_VER_CURRENT);

	start_time=PAPI_get_real_usec();



	/****************************************/
	/* Load the JPEG data 			*/
	/****************************************/

	/* Load an image */

	load_before=PAPI_get_real_usec();
	load_jpeg(argv[1],&image);
	load_after=PAPI_get_real_usec();

	/* Allocate space for output image */
	new_image.x=image.x;
	new_image.y=image.y;
	new_image.depth=image.depth;
	new_image.pixels=(unsigned char *)calloc(image.x*image.y*image.depth,
			sizeof(char));


	int size=image.x*image.y*image.depth*sizeof(char);
	int pitch=image.x*image.depth;


	/*********************/
	/* Initialize OpenCL */
	/*********************/

	/* Get Platforms */
	/* Arguments are: size of the platform array */
	/*                pointer to platform array  */
	/*                number of platforms available */
	err = clGetPlatformIDs(MAX_PLATFORMS,platform,&num_platforms);
	if (err != CL_SUCCESS) {
		printf("Error: Failed to get a platform id! %s\n",
			cl_getErrorString(err));
		return EXIT_FAILURE;
	}


	/* Get the platform info */
	if (num_platforms>MAX_PLATFORMS) num_platforms=MAX_PLATFORMS;

	printf("\nOpenCL platform information, found %d platforms:\n\n",
		num_platforms);

	for(i=0;i<num_platforms;i++) {
		err  = clGetPlatformInfo(platform[i], CL_PLATFORM_NAME,
			sizeof(platform_name[i]), platform_name[i],
			&returned_size);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to get platform info! %s\n",
				cl_getErrorString(err));
			return EXIT_FAILURE;
		}
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_VERSION,
			sizeof(platform_vers[i]), platform_vers[i],
			&returned_size);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to get platform info! %s\n",
				cl_getErrorString(err));
			return EXIT_FAILURE;
		}
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_PROFILE,
			sizeof(platform_prof[i]), platform_prof[i],
			&returned_size);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to get platform info! %s\n",
				cl_getErrorString(err));
			return EXIT_FAILURE;
		}
		err = clGetPlatformInfo(platform[i], CL_PLATFORM_EXTENSIONS,
			sizeof(platform_exts[i]), platform_exts[i],
			&returned_size);
		if (err != CL_SUCCESS) {
			printf("Error: Failed to get platform info! %s\n",
				cl_getErrorString(err));
			return EXIT_FAILURE;
		}

		printf("Platform %d:\n",i);
		printf("\tPlatform name:       %s\n", (char *)platform_name[i]);
		printf("\tPlatform version:    %s\n", (char *)platform_vers[i]);
		printf("\tPlatform profile:    %s\n", (char *)platform_prof[i]);
		printf("\tPlatform extensions: %s\n",
			((char)platform_exts[i][0] != '\0') ?
			(char *)platform_exts[i] : "NONE");
		printf("\n");
	}

	/* Get number of devices */
	printf("-------------------------------------\n");
	printf("Using platform %d (%s) for calculations.\n\n",
		which_platform,platform_name[which_platform]);

	err = clGetDeviceIDs(platform[which_platform], CL_DEVICE_TYPE_ALL,
			MAX_DEVICES, devices, &num_devices);

	if (err != CL_SUCCESS) {
		printf("Failed to collect device list on this platform! %s\n",
			cl_getErrorString(err));
		return EXIT_FAILURE;
	}

	printf("Found %d compute device for this platform.\n",num_devices);


	/* Create a compute context for the requested device */

	device_id=devices[0];

	context = clCreateContext(NULL, 1, &device_id, NULL, NULL, &err);

	if (!context) {
		printf("Error: Failed to create a compute context! %s\n",
			cl_getErrorString(err));
		return EXIT_FAILURE;
	}


	/* Create a command queue */

	/* Note: clCreateCommandQueue was deprecated with OpenCL 2.0	*/
	/*       and should eventually be replaced by               	*/
	/*       clCreateCommandQueueWithProperties()			*/
	commands = clCreateCommandQueue(context, device_id, 0, &err);

	if (!commands) {
		printf("Error: Failed to create a command queue! %s\n",
			cl_getErrorString(err));
		return EXIT_FAILURE;
	}


	/******************************/
	/* Create the combine kernel  */
	/******************************/

	/* arguments are: the context
			how many strings,
			pointer to the strings
			string length array (NULL if they are NUL terminated)
			error return */
	program_combine = clCreateProgramWithSource(context, 1,
			(const char **) &KernelSource_combine, NULL, &err);

	if ((!program_combine) || (err!=CL_SUCCESS)) {
		printf("Error: Failed to create compute program! %s\n",
						cl_getErrorString(err));
		return EXIT_FAILURE;
	}

	/* Build the program */

	/* arguments are:	program,
				num_devices (devices in dev_list)
				dev_list (NULL means build for all)
				options: string to pass to compiler
				pfn_notify: callback when compiling done
				user_data: used with pfn_notify */
	err = clBuildProgram(program_combine, 0, NULL, NULL, NULL, NULL);

	/* Print errors from the kernel compile */
	if (err != CL_SUCCESS) {
    		size_t len;
    		char buffer[2048];
   		printf("Error: Failed to build program executable! %s\n",
						cl_getErrorString(err));
		clGetProgramBuildInfo(program_combine, device_id,
			CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}


	/* Create kernel from the compiled code */

	kernel_combine = clCreateKernel(program_combine,
					"opencl_combine", &err);

	if (!kernel_combine || err != CL_SUCCESS) {
		printf("Error: Failed to create compute kernel %s!\n",
						cl_getErrorString(err));
		exit(1);
	}

	/*******************************/
	/* Create the convolve kernel  */
	/*******************************/

	program_convolve = clCreateProgramWithSource(context, 1,
			(const char **) &KernelSource_convolve, NULL, &err);

	if ((!program_convolve) || (err!=CL_SUCCESS)) {
		printf("Error: Failed to create compute program! %s\n",
						cl_getErrorString(err));
		return EXIT_FAILURE;
	}

	/* Build the program */

	err = clBuildProgram(program_convolve, 0, NULL, NULL, NULL, NULL);

	/* Print errors from the kernel compile */
	if (err != CL_SUCCESS) {
    		size_t len;
    		char buffer[2048];
   		printf("Error: Failed to build program executable! %s\n",
						cl_getErrorString(err));
		clGetProgramBuildInfo(program_convolve, device_id,
			CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
		printf("%s\n", buffer);
		exit(1);
	}

	/* Create kernel from the compiled code */

	kernel_convolve = clCreateKernel(program_convolve, "opencl_convolve", &err);

	if (!kernel_convolve || err != CL_SUCCESS) {
		printf("Error: Failed to create compute kernel %s!\n",
						cl_getErrorString(err));
		exit(1);
	}





	/*************************************/
	/* Allocate space on device          */
	/*************************************/

	cl_mem dev_image;
	cl_mem dev_sobelx;
	cl_mem dev_sobely;
	cl_mem dev_output;
	cl_mem dev_matrix;

	/* Input data, read-only on device */
	dev_image = clCreateBuffer(context, CL_MEM_READ_ONLY,
				sizeof(char) * size, NULL, NULL);
	/* Convolution Matrix, read-only on device */
	dev_matrix = clCreateBuffer(context, CL_MEM_READ_ONLY,
				sizeof(int)*9,NULL,NULL);
	/* SobelX and SobelY on device, read/write on device */
	dev_sobelx = clCreateBuffer(context, CL_MEM_READ_WRITE,
				sizeof(char) * size, NULL, NULL);
	dev_sobely = clCreateBuffer(context, CL_MEM_READ_WRITE,
				sizeof(char) * size, NULL, NULL);
	/* Output is result of combine, write only on device */
	dev_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
				sizeof(char) * size, NULL, NULL);




	/* Copy image data to the device */
	copy_before=PAPI_get_real_usec();
	err  = clEnqueueWriteBuffer(commands, dev_image, CL_TRUE, 0,
			sizeof(char) * size, image.pixels, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error copying! %s\n", cl_getErrorString(err));
		exit(1);
	}
	copy_after=PAPI_get_real_usec();

	/********************************/
	/* Calculate Sobel X		*/
	/********************************/

	convolve_before=PAPI_get_real_usec();

	/* Copy sobelx convolution matrix to device */
	err = clEnqueueWriteBuffer(commands, dev_matrix, CL_TRUE, 0,
					sizeof(int) * 9,
					sobel_x_filter, 0, NULL, NULL);
	if (err != CL_SUCCESS) {
		printf("Error copying! %s\n", cl_getErrorString(err));
		exit(1);
	}

	/* Set the kernel arguments */
	err  = 0;
	err |= clSetKernelArg(kernel_convolve, 0, sizeof(unsigned int), &pitch);
	err |= clSetKernelArg(kernel_convolve, 1, sizeof(unsigned int), &size);
	err |= clSetKernelArg(kernel_convolve, 2, sizeof(cl_mem), &dev_image);
	err |= clSetKernelArg(kernel_convolve, 3, sizeof(cl_mem), &dev_matrix);
	err |= clSetKernelArg(kernel_convolve, 4, sizeof(cl_mem), &dev_sobelx);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments! sobel_x\n");
		exit(1);
	}


	/* Get the size of a thread group on this platform */
	err = clGetKernelWorkGroupInfo(kernel_convolve, device_id,
		CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to retrieve kernel work "
				"group info! %d\n", err);
		exit(1);
	}

	/* Set the size of our problem (number of pixels to calculate) */

	global = size;
	printf("Problem size is %d, dividing up in thread blocks of %d\n",
		 (int)global,local);

	/* arguments are:
			command queue,
			kernel to run,
			number of dimensions of thread/block/grid
			work offset (future: currently always NULL)
			work size: number of elements
			local size: thread split up (NULL means auto)
			num in wait list:
			wait_list: list of events that must finish before run
			events: returns event object for this request
	*/
	err = clEnqueueNDRangeKernel(commands, kernel_convolve,
			1, NULL, &global, NULL, 0, NULL, NULL);

	if (err) {
		printf("Error: Failed to execute kernel (Convolve x)! %s\n",
					cl_getErrorString(err));
		return EXIT_FAILURE;
	}

	/* Finish and synchronize (is this necessary?) */
	clFinish(commands);


	/********************************/
	/* Calculate Sobel Y		*/
	/********************************/

	/* Copy sobely convolution matrix to device */
	err  = clEnqueueWriteBuffer(commands, dev_matrix, CL_TRUE, 0,
						sizeof(int) * 9,
						sobel_y_filter, 0, NULL, NULL);


	/* Set the kernel arguments */

	err  = 0;
	err |= clSetKernelArg(kernel_convolve, 0, sizeof(unsigned int), &pitch);
	err |= clSetKernelArg(kernel_convolve, 1, sizeof(unsigned int), &size);
	err |= clSetKernelArg(kernel_convolve, 2, sizeof(cl_mem), &dev_image);
	err |= clSetKernelArg(kernel_convolve, 3, sizeof(cl_mem), &dev_matrix);
	err |= clSetKernelArg(kernel_convolve, 4, sizeof(cl_mem), &dev_sobely);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments! sobel_y %d\n");
		exit(1);
	}


	/* Queue up the convolve kernel again, this time sobel_y */

	err = clEnqueueNDRangeKernel(commands, kernel_convolve, 1, NULL,
		&global, NULL, 0, NULL, NULL);

	if (err) {
		printf("Error: Failed to execute kernel (Convolve Y)!\n");
		return EXIT_FAILURE;
	}

	clFinish(commands);

	convolve_after=PAPI_get_real_usec();


	/********************************/
	/* Combine to form output	*/
	/********************************/

	combine_before=copy_after;

	/* Set the kernel arguments */
	err  = 0;
	err |= clSetKernelArg(kernel_combine, 0, sizeof(unsigned int), &size);
	err |= clSetKernelArg(kernel_combine, 1, sizeof(cl_mem), &dev_sobelx);
	err |= clSetKernelArg(kernel_combine, 2, sizeof(cl_mem), &dev_sobely);
	err |= clSetKernelArg(kernel_combine, 3, sizeof(cl_mem), &dev_output);

	if (err != CL_SUCCESS) {
		printf("Error: Failed to set kernel arguments! combine\n");
		exit(1);
	}

	/* Queue up the convolve kernel again, this time comhine */

	err = clEnqueueNDRangeKernel(commands, kernel_combine, 1, NULL,
		&global, NULL, 0, NULL, NULL);

	if (err) {
		printf("Error: Failed to execute kernel (combine)!\n");
		return EXIT_FAILURE;
	}

	clFinish(commands);

	combine_after=PAPI_get_real_usec();

	copy2_before=PAPI_get_real_usec();

	/* Copy the results back to host */
	err = clEnqueueReadBuffer( commands, dev_output, CL_TRUE, 0,
		sizeof(char) * size, new_image.pixels, 0, NULL, NULL );

	if (err != CL_SUCCESS) {
		printf("Error: Failed to read output array! %d\n", err);
		exit(1);
	}

	copy2_after=PAPI_get_real_usec();






	/* Write data back out to disk */
	store_before=PAPI_get_real_usec();
	store_jpeg("out.jpg",&new_image);
	store_after=PAPI_get_real_usec();

	/* Print timing results */
	printf("PAPI timing results:\n");
	printf("\tLoad time: %lld\n",load_after-load_before);
        printf("\tConvolve time: %lld\n",convolve_after-convolve_before);
	printf("\tCopy time: %lld\n",(copy_after-copy_before)+
				(copy2_after-copy2_before));
        printf("\tCombine time: %lld\n",combine_after-combine_before);
        printf("\tStore time: %lld\n",store_after-store_before);
	printf("\tTotal time = %lld\n",store_after-start_time);

	return 0;
}
