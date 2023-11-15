#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

#define BLOCK_SIZE 32  //block size ,each thread to calculate each bloc

void initial(float *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() / double(RAND_MAX);
    }
}

__global__ void  matrixMultiplyShared(float *A, float *B, float *C,
	int numARows, int numAColumns, int numBRows, int numBColumns, int numCRows, int numCColumns)
{
	//@@ Insert code to implement matrix multiplication here
	//@@ You have to use shared memory for this MP
    	__shared__ float sharedM[BLOCK_SIZE][BLOCK_SIZE];
	__shared__ float sharedN[BLOCK_SIZE][BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tx = threadIdx.x;
	int ty = threadIdx.y;


	int row = by * BLOCK_SIZE + ty;
	int col = bx * BLOCK_SIZE + tx;


	float Csub = 0.0;

	for (int i = 0; i < (int)(ceil((float)numAColumns / BLOCK_SIZE)); i++)
	{
		// printf("block.x=%d,block.y=%d,threadIdx.x=%d,threadIdx.y=%d,row=%d,col=%d,sharedM[%d][%d]=A[%d],A的值：%f,sharedN[%d][%d]=B[%d],B的值：%f\n",
		// 	blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, row, col,
		// 	threadIdx.y, threadIdx.x, row*numAColumns + i * BLOCK_SIZE + tx, A[row*numAColumns + i * BLOCK_SIZE + tx],
		// 	threadIdx.y, threadIdx.x, (i*BLOCK_SIZE + ty)*numBColumns + col, B[(i*BLOCK_SIZE + ty)*numBColumns + col]);

		if (i*BLOCK_SIZE + tx < numAColumns && row < numARows)
			sharedM[ty][tx] = A[row*numAColumns + i * BLOCK_SIZE + tx];
		else
			sharedM[ty][tx] = 0.0;

		if (i*BLOCK_SIZE + ty < numBRows && col < numBColumns)
			sharedN[ty][tx] = B[(i*BLOCK_SIZE + ty)*numBColumns + col];
		else
			sharedN[ty][tx] = 0.0;
		__syncthreads();


		for (int j = 0; j < BLOCK_SIZE; j++)
			Csub += sharedM[ty][j] * sharedN[j][tx];
		__syncthreads();
	}


	if (row < numCRows && col < numCColumns)
		C[row*numCColumns + col] = Csub;
}

int main(int argc, char **argv){
    // A: M * K, B: K * N, C: M * N
    if (argc < 4) {
        cout << "Usage: ./matrixMultiplyCPU <M> <K> <N>" << endl;
        return -1;
    }

    int M = stoi(argv[1]);
    int K = stoi(argv[2]);
    int N = stoi(argv[3]);

    // size
    int Axy = M * K;
    int Bxy = K * N;
    int Cxy = M * N;

    // Matrix
    float* A, *B, *C;
    A = (float*)malloc(Axy * sizeof(float ));
    B = (float*)malloc(Bxy * sizeof(float ));
    C = (float*)malloc(Cxy * sizeof(float ));

    // initialize A, B
    initial(A, Axy);
    initial(B, Bxy);


    // gpu allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, Axy * sizeof(float));
    cudaMalloc((void**)&d_B, Bxy * sizeof(float));
    cudaMalloc((void**)&d_C, Cxy * sizeof(float));

    cudaMemcpy(d_A, A, Axy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, Bxy * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using multiplicateMatrixOnDevice \n");
	printf("------------------------------------------------------------------------------------\n");

    // kernel operation
    int dimX = 2;
    int dimY = 2;
    dim3 block(dimX, dimY);
	dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);  // 

    cudaEvent_t gpustart, gpustop;
	float elapsedTime = 0.0;

	elapsedTime = 0.0;
	cudaEventCreate(&gpustart);
	cudaEventCreate(&gpustop);
	cudaEventRecord(gpustart, 0);
	matrixMultiplyShared << < grid, block >> > (d_A, d_B, d_C, M, K, K, N, M, N);
	//	printf("   multiplicateMatrixOnDevice<<<(%d,%d),(%d,%d)>>>", grid.x, grid.y, block.x, block.y);
	cudaDeviceSynchronize();
	cudaEventRecord(gpustop, 0);
	cudaEventSynchronize(gpustop);

	cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
	cudaEventDestroy(gpustart);
	cudaEventDestroy(gpustop);

	cudaMemcpy(C, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fs\n",
		M, N, grid.x, grid.y, block.x, block.y, elapsedTime / 1000);
    return 0;
}
