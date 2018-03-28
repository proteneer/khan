// kernel_example.h
#ifndef ANI_OP_H_
#define ANI_OP_H_

template <typename Device, typename T>
struct AniFunctor {
  void operator()(const Device& d, int size, const T* in, T* out);
};

#if GOOGLE_CUDA
// Partially specialize functor for GpuDevice.
// (YTZ): A dd additional bookkeeping memory here
template <typename Eigen::GpuDevice, typename T>
struct AniFunctor {
  void operator()(const Eigen::GpuDevice& d, int size, const T* in, T* out);
};
#endif

#endif ANI_OP_H_