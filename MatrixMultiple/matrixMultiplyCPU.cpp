//
// Created by shengchen on 11/14/23.
//
#include <iostream>
#include <ctime>

using namespace std;

void initial(float *array, int size)
{
    for (int i = 0; i < size; i++)
    {
        array[i] = rand() / double(RAND_MAX);
    }
}

void matrixMultiplyCPU(float* A, float* B, float* C, int M, int K, int N) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            float sum = 0;
            for (int k = 0; k < K; k++) {
                sum += A[i*K+k] * B[k*N+j];
            }
            C[i*N+j] = sum;
        }
    }
}

int main(int argc, char *argv[]){
    clock_t start = 0, finish = 0;
    float time;

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

    // compute matrix multiply
    start = clock();
    matrixMultiplyCPU(A, B, C, M, K, N);
    finish = clock();
    time = (float)(finish - start) / CLOCKS_PER_SEC;

    printf("\n");
    printf("------------------------------------------------------------------------------------\n");
    printf("Computing matrix product using matrixMultiplyCPUOnHost \n");
    printf("------------------------------------------------------------------------------------\n");

    printf("Matrix_hostRef: (%d×%d)  CPU运行时间为：%lfs\n", M, N, time);
}