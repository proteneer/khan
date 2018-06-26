#include <vector>

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/types.h"
#include "memcpy.h"

using CPUDevice = Eigen::ThreadPoolDevice;

template<>
void device_memcpy<CPUDevice>(float *dst, const std::vector<float> &src) {
  if(src.size() > 0) {
    memcpy(dst, &src[0], src.size()*sizeof(float));    
  }
};

template void device_memcpy<CPUDevice>(float *dst, const std::vector<float> &src);