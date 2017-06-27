#include <stdio.h>
__global__ void global_reduce_kernel(float * d_out, float * d_in)
{
  int my_id = threadIdx.x + blockDim.x * blockIdx.x;
  int tid = threadIdx.x;

  // Reduction in global memory - Divides the active regions into half.
  for (unsigned int i = blockDim.x / 2; i > 0; i >>= 1){  // >>= bitwise right shift assignment 
    // 1024 -> 512 + 512 : Add elements in its first half, to its second half, writing back to first half; recurse
    if(tid < i){
      d_in[my_id] += d_in[my_id + i];
    }
    __syncthreads();  // All stages of adds are done here
  
  }
  // Only one elements remains here, write to global memory
  if(tid == 0){
    d_out[blockIdx.x] = d_in[my_id]
  }
}