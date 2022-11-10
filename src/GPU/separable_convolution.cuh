#ifndef HARRIS_CORNER_CUDA_SEPARABLE_CONVOLUTION_CUH
#define HARRIS_CORNER_CUDA_SEPARABLE_CONVOLUTION_CUH

#include "../render.hpp"

namespace gpu
{
    enum class KernelType
    {
        GAUSSIAN = 0,
        GAUSSIAN_DERIV = 1
    };

    const double h_Gauss[KERNEL_LENGTH] = {
        0.011109, 0.13533528, 0.60653066, 1, 0.60653066, 0.13533528, 0.011109
    };
    const double h_GaussDeriv[KERNEL_LENGTH] = {
        0.03332699, 0.27067057, 0.60653066, 0, -0.60653066, -0.27067057, -0.03332699
    };

    extern "C" void setGaussianKernel();
    extern "C" void setDerivGaussianKernel();

    extern "C" void convolutionRowsGPU(double *d_Dst, size_t dstPitch, const double *d_Src, size_t srcPitch,
                                       int imageW, int imageH, KernelType kernelType);

    extern "C" void convolutionColumnsGPU(double *d_Dst, size_t dstPitch, const double *d_Src, size_t srcPitch,
                                          int imageW, int imageH, KernelType kernelType);
}

#endif // HARRIS_CORNER_CUDA_SEPARABLE_CONVOLUTION_CUH
