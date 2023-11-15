#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <iostream>

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "cublas_v2.h"

using namespace std;

#define BLOCK_SIZE 32  //block size ,each thread to calculate each bloc

void initial(float *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() / double(RAND_MAX);
    }
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
    float* h_A, *h_B, *h_C;
    h_A = (float*)malloc(Axy * sizeof(float ));
    h_B = (float*)malloc(Bxy * sizeof(float ));
    h_C = (float*)malloc(Cxy * sizeof(float ));

    // initialize A, B
    initial(h_A, Axy);
    initial(h_B, Bxy);


    // gpu allocation
    float *d_A, *d_B, *d_C;
    cudaMalloc((void**)&d_A, Axy * sizeof(float));
    cudaMalloc((void**)&d_B, Bxy * sizeof(float));
    cudaMalloc((void**)&d_C, Cxy * sizeof(float));

    cudaMemcpy(d_A, h_A, Axy * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, Bxy * sizeof(float), cudaMemcpyHostToDevice);

    printf("\n\n");
	printf("------------------------------------------------------------------------------------\n");
	printf("Computing matrix product using multiplicateMatrixOnDevice \n");
	printf("------------------------------------------------------------------------------------\n");

	int dimx = 2;
	int dimy = 2;
	dim3 block(dimx, dimy);
	dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

    // cublass
    cublasStatus_t status;
    cublasHandle_t handle;
    cublasCreate(&handle);

    float elapsedTime = 0.0;
    cudaEvent_t gpustart, gpustop;
    cudaEventCreate(&gpustart);
    cudaEventCreate(&gpustop);
    cudaEventRecord(gpustart, 0);

    float a = 1, b = 0;
    cublasSgemm(
        handle,
        CUBLAS_OP_T,   //矩阵A的属性参数，转置，按行优先
        CUBLAS_OP_T,   //矩阵B的属性参数，转置，按行优先
        M,          //矩阵A、C的行数
        N,          //矩阵B、C的列数
        K,          //A的列数，B的行数，此处也可为B_ROW,一样的
        &a,             //alpha的值
        d_A,            //左矩阵，为A
        K,          //A的leading dimension，此时选择转置，按行优先，则leading dimension为A的列数
        d_B,            //右矩阵，为B
        N,          //B的leading dimension，此时选择转置，按行优先，则leading dimension为B的列数
        &b,             //beta的值
        d_C,            //结果矩阵C
        M           //C的leading dimension，C矩阵一定按列优先，则leading dimension为C的行数
    );

    cudaMemcpy(h_C, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
	cudaDeviceSynchronize();
	cudaEventRecord(gpustop, 0);
	cudaEventSynchronize(gpustop);

	cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
	cudaEventDestroy(gpustart);
	cudaEventDestroy(gpustop);

	cudaMemcpy(h_C, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
	printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fs\n",
		M, N, grid.x, grid.y, block.x, block.y, elapsedTime / 1000);

    cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(h_C);

	cudaDeviceReset();
    return 0;
}
