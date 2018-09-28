#include "tensorflow/core/framework/op_kernel.h"
#include "memcpy.h"

using GPUDevice = Eigen::GpuDevice;

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


template<>
void device_memcpy<GPUDevice>(float *dst, const std::vector<float> &src) {
  if(src.size() > 0) {
    gpuErrchk(cudaMemcpy(dst, &src[0], src.size()*sizeof(float), cudaMemcpyHostToDevice));
  }
}

template void device_memcpy<GPUDevice>(float *dst, const std::vector<float> &src);