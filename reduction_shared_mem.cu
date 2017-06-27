/* For detailed documentation, see reduction_global_mem.cu, only difference is
with the usage of shared memory instead of global 
 */
#include <stdio.h>
__global__ void shmem_reduce_kernel(float * d_out, const float * d_in){
  // Shared Data (sdata) is allocated in kernel call as 3rd arg
  extern __shared__ float sdata[];
  int myId = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;
  // load  all the data from global memory to shared memory
  sdata[tid] = d_in[myId];
  __syncthreads();  //Make sure the entire block is loaded
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1){
    if(tid < i){
      sdata[tid] += sdata[tid + i];
    }
    __syncthreads();
  }
  if(tid == 0){
    d_out[blockIdx.x] = sdata[0]
  }
}

