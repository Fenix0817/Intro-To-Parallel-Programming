#include <stdio.h>
#include "gputimer.h" // GPU Timer

#define NUM_THREADS 1000000
#define ARRAY_SIZE 10

#define BLOCK_WIDTH 1000

void print_array(int *array, int size){
    printf("{");
    for (int i = 0; i<size; i++){
        printf("%d ", array[i])
    }
    printf("}")
}

__global__ void increment_naive(int *g_array){
    // Map thread indexes
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    // Consecutive threads will increment consecutive elements, wrapping at ARRAY_SIZE
    i = i % ARRAY_SIZE;
    g_array[i] = g_array[i] + 1
}

// Driver Program
int main(int argc, char **argv){
    GpuTimer timer;
    printf("%d Total threads in %d Blocks writing inot %d Array elements",NUM_THREADS, NUM_THREADS/BLOCK_WIDTH, ARRAY_SIZE);
    // Host memory
    int h_array[ARRAY_SIZE];
    const int ARRAY_BYTES = ARRAY_SIZE * sizeof(int);
    // Device memory
    int * d_array;
    cudaMalloc((void **) &d_array, ARRAY_BYTES);
    cudaMemset((void **) &d_array, 0, ARRAY_BYTES); // Zero out device memory
    // Kernel Launch
    timer.Start();
    increment_naive<<<NUM_THREADS/BLOCK_WIDTH, BLOCK_WIDTH>>>(d_array);
    timer.Stop();
    // Copy back result to host memory
    cudaMemcpy(h_array, d_array, ARRAY_BYTES, cudaMemcpyDeviceToHost);
    print_array(h_array, ARRAY_SIZE);
    printf("Time elapsed is %g ms \n", timer.Elapsed());
    // Free allocated memory
    cudaFree(d_array);
    return 0;
    
}