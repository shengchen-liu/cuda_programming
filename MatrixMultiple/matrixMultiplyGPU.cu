#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

__global__
void multiplyMatrixOnDevice(float* d_A, float* d_B, float* d_C, int M, int K, int N) {
    int ix = threadIdx.x + blockDim.x * blockIdx.x; // row number
	int iy = threadIdx.y + blockDim.y*blockIdx.y;//col number

	if (ix < N_p && iy < M_p)
	{
		float sum = 0;
		for (int k = 0; k < K_p; k++)
		{
			sum += array_A[iy*K_p + k] * array_B[k*N_p + ix];
		}
		array_C[iy*N_p + ix] = sum;
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

    // kernel operation
    int dimX = 2;
    int dimY = 2;
    dim3 block(dimX, dimY);
	dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);  // 

	multiplyMatrixOnDevice<<<grid,block>>> (d_A, d_B, d_C, M, K, N);

    // copy to host
    cudaMe
    ss
    return 0;
}