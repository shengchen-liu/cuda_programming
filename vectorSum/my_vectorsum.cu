/*
 * A + B = C vector summation
 * */
#include <iostream>
#include <cstdlib>
#include <sys/time.h>
#include <cuda_runtime.h>

using namespace std;

__global__
void vecAddKernel(float* A_d, float* B_d, float* C_d, int n) {
  int i = threadIdx.x + blockDim.x * blockIdx.x;
  if (i < n) C_d[i] = A_d[i] + B_d[i];
}

int main(int argc, char *argv[]) {
  int n = atoi(argv[1]);
  cout << n << endl;

  size_t size = n * sizeof(float);

  // host memory
  float *a = (float *)malloc(size);
  float *b = (float *)malloc(size);
  float *c = (float *)malloc(size);

  for (int i = 0; i < n; ++i) {
    float af = rand() / double(RAND_MAX);
    float bf = rand() / double(RAND_MAX);
    a[i] = af;
    b[i] = bf;
  }

  // cuda memory
  float *da = NULL;
  float *db = NULL;
  float *dc = NULL;

  // This is a typecast. It indicates that the type of the pointer being passed is a pointer to void,
  // which is a generic type in C/C++ that can be used to represent any type.
  // In CUDA, memory allocation functions, like cudaMalloc, expect a pointer to void as their argument.
  cudaMalloc((void **)&da, size);
  cudaMalloc((void **)&db, size);
  cudaMalloc((void **)&dc, size);

  cudaMemcpy(da, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(db, a, size, cudaMemcpyHostToDevice);
  cudaMemcpy(dc, a, size, cudaMemcpyHostToDevice);

  // kernel operate
  struct timeval t1, t2;

  int threadPerBlock = 256;
  gettimeofday(&t1, NULL);
  vecAddKernel<<<ceil(n / threadPerBlock), threadPerBlock>>>(da, db, dc, n);
  gettimeofday(&t2, NULL);

  // copy results to host
  cudaMemcpy(c,dc,size,cudaMemcpyDeviceToHost);

  double timeuse = (t2.tv_sec - t1.tv_sec) + (double)(t2.tv_usec - t1.tv_usec)/1000000.0;
  cout << timeuse << endl;

  cudaFree(da);
  cudaFree(db);
  cudaFree(dc);

  free(a);
  free(b);
  free(c);
  return 0;
}