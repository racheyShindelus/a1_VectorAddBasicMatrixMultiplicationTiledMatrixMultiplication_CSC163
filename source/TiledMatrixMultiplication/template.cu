#include <cuda_runtime.h> 
#include <device_launch_parameters.h> 
#include <wb.h>
#include <fstream>

#define TILE_WIDTH 16 	//do not change this value

#define wbCheck(stmt)                                                     \
  do {                                                                    \
    cudaError_t err = stmt;                                               \
    if (err != cudaSuccess) {                                             \
      wbLog(ERROR, "Failed to run stmt ", #stmt);                         \
      wbLog(ERROR, "Got CUDA error ...  ", cudaGetErrorString(err));      \
      return -1;                                                          \
    }                                                                     \
  } while (0)

// Compute C = A * B
__global__ void matrixMultiplyShared(float *A, float *B, float *C,
                                     int numARows, int numAColumns,
                                     int numBColumns) {
    //@@ Insert code to implement tiled matrix multiplication here
    //@@ You have to use shared memory to write this kernel
    __shared__ float ds_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_B[TILE_WIDTH][TILE_WIDTH];

    int row = blockIdx.y * TILE_WIDTH + threadIdx.y;
    int col = blockIdx.x * TILE_WIDTH + threadIdx.x;

    float value = 0.0f;
    int numTiles = (numAColumns + TILE_WIDTH - 1) / TILE_WIDTH;

    for (int t = 0; t < numTiles; t++) {
        int A_col = t * TILE_WIDTH + threadIdx.x;
        int B_row = t * TILE_WIDTH + threadIdx.y;

        // Load tiles into shared memory
        // tile a
        ds_A[threadIdx.y][threadIdx.x] = (row < numARows&& A_col < numAColumns) ? A[row * numAColumns + A_col] : 0.0f;
        // tile b
        ds_B[threadIdx.y][threadIdx.x] = (B_row < numAColumns&& col < numBColumns) ? B[B_row * numBColumns + col] : 0.0f;

        __syncthreads();            // says an error but vscode just doesn't detect the CUDA

        // Multiply the two tiles
        for (int k = 0; k < TILE_WIDTH; k++) {
            value += ds_A[threadIdx.y][k] * ds_B[k][threadIdx.x];
        }
        __syncthreads();            // says an error but vscode just doesn't detect the CUDA
    }

    if (row < numARows && col < numBColumns) {
        C[row * numBColumns + col] = value;
    }
}

int main(int argc, char **argv) {
  wbArg_t args;
  float *hostA; // The A matrix
  float *hostB; // The B matrix
  float *hostC; // The output C matrix
  float *deviceA;
  float *deviceB;
  float *deviceC;
  int numARows;    // number of rows in the matrix A
  int numAColumns; // number of columns in the matrix A
  int numBRows;    // number of rows in the matrix B
  int numBColumns; // number of columns in the matrix B
  int numCRows;    // number of rows in the matrix C (you have to set this)
  int numCColumns; // number of columns in the matrix C (you have to set
                   // this)
  
  hostC = NULL;

  args = wbArg_read(argc, argv);

  wbTime_start(Generic, "Importing data and creating memory on host");
  hostA = (float *)wbImport(wbArg_getInputFile(args, 0), &numARows,
                            &numAColumns);
  hostB = (float *)wbImport(wbArg_getInputFile(args, 1), &numBRows,
                            &numBColumns);
  //@@ Set numCRows and numCColumns
  numCRows    = numARows;
  numCColumns = numBColumns;

  //@@ Allocate the hostC matrix
  hostC = (float*)malloc(numCRows * numCColumns * sizeof(float));

  wbTime_stop(Generic, "Importing data and creating memory on host");

  wbLog(TRACE, "The dimensions of A are ", numARows, " x ", numAColumns);
  wbLog(TRACE, "The dimensions of B are ", numBRows, " x ", numBColumns);

  wbTime_start(GPU, "Allocating GPU memory.");
  //@@ Allocate GPU memory here
  wbCheck(cudaMalloc((void**)&deviceA, numARows * numAColumns * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceB, numBRows * numBColumns * sizeof(float)));
  wbCheck(cudaMalloc((void**)&deviceC, numCRows * numCColumns * sizeof(float)));

  wbTime_stop(GPU, "Allocating GPU memory.");

  wbTime_start(GPU, "Copying input memory to the GPU.");
  //@@ Copy memory to the GPU here
  wbCheck(cudaMemcpy(deviceA, hostA, numARows * numAColumns * sizeof(float), cudaMemcpyHostToDevice));
  wbCheck(cudaMemcpy(deviceB, hostB, numBRows * numBColumns * sizeof(float), cudaMemcpyHostToDevice));

  wbTime_stop(GPU, "Copying input memory to the GPU.");

  // @@ Initialize the grid and block dimensions here
  dim3 dimBlock(TILE_WIDTH, TILE_WIDTH, 1);
  dim3 dimGrid((numCColumns + TILE_WIDTH - 1) / TILE_WIDTH, (numCRows + TILE_WIDTH - 1) / TILE_WIDTH, 1);

  wbTime_start(Compute, "Performing CUDA computation");
  //@@ Launch the GPU Kernel here
  // states "expects an expression" but still works CUDA
  matrixMultiplyShared <<< dimGrid, dimBlock >>> (deviceA, deviceB, deviceC, numARows, numAColumns, numBColumns);
  cudaDeviceSynchronize();

  wbTime_stop(Compute, "Performing CUDA computation");
    
  wbTime_start(Copy, "Copying output memory to the CPU");
  //@@ Copy the GPU memory back to the CPU here
  wbCheck(cudaMemcpy(hostC, deviceC, numCRows * numCColumns * sizeof(float), cudaMemcpyDeviceToHost));

  wbTime_stop(Copy, "Copying output memory to the CPU");

  std::ofstream outFile("myoutput.raw", std::ios::binary);

  wbTime_start(GPU, "Freeing GPU Memory");
  //@@ Free the GPU memory here
  cudaFree(deviceA);
  cudaFree(deviceB);
  cudaFree(deviceC);

  wbTime_stop(GPU, "Freeing GPU Memory");

  wbSolution(args, hostC, numCRows, numCColumns);


  free(hostA);
  free(hostB);
  free(hostC);

  return 0;
}
