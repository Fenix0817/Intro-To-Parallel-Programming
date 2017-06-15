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

/* 
 * Shared Memory Usage
 */

__global__ void use_shared_memory_GPU(float * array)
{
    //local variables, that are private to each thread
    int i, index = threadIdx.x;
    float average, sum = 0.0f;
    // Shared variable that is visible to all the threads in the threadblock, and has the same lifetime as the thread block
    __shared__ float sh_array[128]; // Hard coding 128 elements
    // Copy data from global memory -> array to shared memory -> sh_array, such that each thread is responsible for copying each element (Hard-coded to have 128 threads)
    sh_array[index] = array[index];
    __syncthreads(); // Sync Threads Barrier: Allow all the threads to complete the above copying process
    // Find running average of all elements in the array to its left
    for(i=0; i<index; i++){
        sum += sh_array[i];
    }
    average = sum / (index + 1.0f);
    // Replace the current value with avg, if avg is greater
    if (array[index] > average){
        array[index] = average;
    }
    // Shared memory has a life-span of the duration of thread execution, the following code snippet has no effect
    sh_array[index] = 3.14; 
}

// Driver Program
int main(int argc, char **argv)
{
    /*
     * Local Memory
     */ 
    use_local_memory_GPU<<<1, 128>>>(2.0f);
    
    /*
     * Global Memory
     */ 
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
    
    /*
     * Shared memory
     */ 
    use_shared_memory_GPU<<<1, 128>>(d_array);
    cudaMemcpy((void *)h_array, (void *)d_array, sizeof(float) * 128, cudaMemcpyDeviceToHost);
    return 0;
}