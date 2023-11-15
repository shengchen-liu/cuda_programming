#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

__global__
void multiplyMatrixOnDevice(float* d_A, float* d_B, float* d_C, int M, int K, int N) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // idx along x
	int iy = threadIdx.y + blockDim.y * blockIdx.y; // idx along y

	if (ix < N && iy < M)
	{
		float sum = 0;
		for (int k = 0; k < K; k++)
		{
			sum += d_A[iy*K + k] * d_B[k*N + ix];
		}
		d_C[iy*N + ix] = sum;
	}
}

void initial(float *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() / double(RAND_MAX);
    }
}

int main(int argc, char* argv[]) {
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
	cudaEventCreate(&gpustart);
	cudaEventCreate(&gpustop);
	cudaEventRecord(gpustart, 0);

	multiplyMatrixOnDevice<<<grid,block>>> (d_A, d_B, d_C, M, K, N);

	cudaDeviceSynchronize();
	cudaEventRecord(gpustop, 0);
	cudaEventSynchronize(gpustop);

	cudaEventElapsedTime(&elapsedTime, gpustart, gpustop);
	cudaEventDestroy(gpustart);
	cudaEventDestroy(gpustop);

    // copy to host
    cudaMemcpy(C, d_C, Cxy * sizeof(float), cudaMemcpyDeviceToHost);
    
	printf("Matrix_deviceRef: (%d×%d)  <<<(%d,%d),(%d,%d)>>>  GPU运行时间为：%fs\n",
			M, N, grid.x, grid.y, block.x, block.y, elapsedTime / 1000);

    return 0;
}