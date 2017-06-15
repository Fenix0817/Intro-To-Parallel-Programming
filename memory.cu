// Usage of different memories in cuda

/*
**Local Memory Usage**
*/

__global__ void use_local_memory_GPU(float in)
{
    float f; // variable "f" is in local memory, private to each thread
    f = in; // parameter "in" is in local memory, private to each thread
    
}

int main(int argc, char **argv)
{
    //Kernel Call
    use_local_memory_GPU<<1, 128>>(2.0f);
}