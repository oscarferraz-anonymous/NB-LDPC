 //---------------------------------------------------------------
 // Trabalho PrÃ¡ctico NÂº4 - CUDA I - CHAD
 // Óscar Ferraz
 // 2018/2019
 // --------------------------------------------------------------
 
 
// nvcc -o vecAdd vecAdd.cu -I /usr/local/cuda-9.1/samples/common/inc


#include <stdio.h>
#include <time.h>
#include <cuda_runtime.h>
//#include <helper_cuda.h>



/**
 * CUDA Kernel Device code
 *
 * Computes the vector addition of A and B into C. The 3 vectors have the same
 * number of elements numElements.
 * 
 */
 
 __global__ void histo(unsigned int * hist, unsigned char * Image,int width1, int height1){
	
	__shared__ unsigned int histo_s[256];
	if (threadIdx.x < 256)
		histo_s[threadIdx.x]=0;
	__syncthreads();
	
	int i=threadIdx.x +blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;
	while (i < width1*height1){
		atomicAdd( &(histo_s[Image[i]]), 1);
		i += stride;
	}
	__syncthreads();
	if (threadIdx.x < 256)
		atomicAdd( &(hist[threadIdx.x]),histo_s[threadIdx.x]);
}
/**
 * Host main routine
 */
int
main(void)
{
	int width=813; 
	int height=707;
	int numElements=width*height;
	printf("[Vector addition of %d elements]\n", numElements);
	
	// Error code to check return values for CUDA calls
    cudaError_t err = cudaSuccess;
    
    size_t size=width*height*sizeof(unsigned char);
    size_t size_hist=256*sizeof(unsigned int);
    
    // Allocate the host input vector image
    unsigned char *h_image=(unsigned char *)malloc(size);
    // Allocate the host input vector igrey
    unsigned int *h_hist=(unsigned int *)malloc(size_hist);
    
    // Verify that allocations succeeded
    if (h_image==NULL||h_hist==NULL)
    {
        fprintf(stderr, "Failed to allocate horgbImagest vectors!\n");
        exit(EXIT_FAILURE);
    }
    
	// reading file
	FILE *fp;
    unsigned char ch;
    fp = fopen("model_result.ppm", "r");

    if(fp == NULL)
    {
        printf("Error in opening the image\n");
        fclose(fp);
        return(0);
    }

    printf("Successfully opened the image file\n");
	int k=0;
	int i=0;
    while((ch = fgetc(fp)) != EOF)
    {
		
		if(k>=3){
			h_image[i]=ch;
			i++;
		}
		if(ch == '\n')
			k++;
			
        if(i>=813*707)
			break;
    }
    
    printf("reading completed\n");
    fclose(fp);
    
    
    // Allocate the device input vector image
    unsigned char *d_image=NULL;
    err=cudaMalloc((void **)&d_image, size);
    if(err!=cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector image (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
        // Allocate the device input vector grey
    unsigned int *d_hist=NULL;
    err=cudaMalloc((void **)&d_hist, size_hist);
    if(err!=cudaSuccess)
    {
        fprintf(stderr, "Failed to allocate device vector grey (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy the host input vectors A and B in host memory to the device input vectors in device memory
    printf("Copy input image from the host memory to the CUDA device\n");
    err=cudaMemcpy(d_image, h_image, size, cudaMemcpyHostToDevice);
    if(err!=cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector image from host to device (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
	
	// [!!] Define the number of threads per block and blocks per grid
    int threadsPerBlock =1024;
    int blocksPerGrid = 1;
    printf("CUDA kernel launch with %d blocks of %d threads\n", blocksPerGrid, threadsPerBlock);
    
    struct timespec start, end;
	clock_gettime(CLOCK_MONOTONIC, &start);
     // [!!] Launch the Vector Add CUDA Kernel with the respective input parameters
    histo<<<blocksPerGrid, threadsPerBlock>>>(d_hist, d_image, width, height);
     clock_gettime(CLOCK_MONOTONIC, &end);
    err=cudaGetLastError();
    if(err!=cudaSuccess)
    {
        fprintf(stderr, "Failed to launch colorToGreyScaleConvertion kernel (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Copy the device result vector in device memory to the host result vector in host memory
    printf("Copy output data from the CUDA device to the host memory\n");
    err=cudaMemcpy((unsigned int *)h_hist, (unsigned int *)d_hist, size_hist, cudaMemcpyDeviceToHost);
    if(err!=cudaSuccess)
    {
        fprintf(stderr, "Failed to copy vector grey from device to host (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    

//=====================================================
// write result
//=====================================================
    /*fp = fopen("result_cuda.ppm", "w");
    fprintf(fp, "P6\n813 707\n255\n");
    for(i=0;i<813*707;i++){
		fputc(h_grey[i], fp);
		fputc(h_grey[i], fp);
		fputc(h_grey[i], fp);
	}
    fclose(fp);*/
	
	double initialTime=(start.tv_sec*1e3)+(start.tv_nsec*1e-6);
	double finalTime=(end.tv_sec*1e3)+(end.tv_nsec*1e-6);
	printf("TIme:\t%f ms\n", (finalTime - initialTime));
	
	 // Free device global memory
    err=cudaFree(d_image);
    if(err!=cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector image(error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    err=cudaFree(d_hist);
    if(err!=cudaSuccess)
    {
        fprintf(stderr, "Failed to free device vector grey (error code %s)!\n", cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
    
    // Free host memory
    free(h_image);
    free(h_hist);
	printf("Done\n");
    return 0;
}
