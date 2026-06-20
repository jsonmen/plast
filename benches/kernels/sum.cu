extern "C" __global__ void sum_tokens(const unsigned int *input,
                                      unsigned long long *global_sum,
                                      int num_elements) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  // Shared memory block allocation for block-level warp reduction passes
  __shared__ unsigned long long sdata[256];

  unsigned long long local_sum = 0;
  if (idx < num_elements) {
    local_sum = (unsigned long long)input[idx];
  }

  sdata[threadIdx.x] = local_sum;
  __syncthreads();

  // Perform standard in-block unrolled warp reduction
  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (threadIdx.x < s) {
      sdata[threadIdx.x] += sdata[threadIdx.x + s];
    }
    __syncthreads();
  }

  // Atomically stream the aggregate block summation value out to VRAM
  if (threadIdx.x == 0 && sdata[0] > 0) {
    atomicAdd(global_sum, sdata[0]);
  }
}
