// Usage of different memories in cuda

/*
 * Local Memory Usage**
 */ 
__global__ void use_local_memory_GPU(float in)
{
    float f; // variable "f" is in local memory, private to each thread
    f = in; // parameter "in" is in local memory, private to each thread
    
}

/* 
 * Global memory usage
 */ 
__global__ void use_global_memory_GPU(float *array)
{
    // *array points to a global memory that we have allocated somewhere else 
    array[threadIdx.x] = 2.0f * (float) threadIdx.x; // Double the number in array pointing to the index at that location
}

// Driver Program
int main(int argc, char **argv)
{
    //Kernel Call- Local Memory
    use_local_memory_GPU<<<1, 128>>>(2.0f);
    
    // Global Memory
    float h_array[128]; // Allocate an array on Host memory
    float *d_array; // Pointer that is used to point to the global memory allocated on Device
    // Allocate memory on the device using cudaMalloc
    cudaMalloc((void **) &d_array, sizeof(float) * 128);
    // Copy data from host memory to device memory using cudaMemcpy
    cudaMemcpy((void *)d_array, (void *)h_array, sizeof(float) * 128, cudaMemcpyHostToDevice);
    // Launch Kernel
    use_global_memory_GPU<<<1, 128>>>(d_array);
    // Copy back output from device to host
    cudaMemcpy((void *)h_array, (void *)d_array, sizeof(float) * 128, cudaMemcpyDeviceToHost);
    
    return 0;
}