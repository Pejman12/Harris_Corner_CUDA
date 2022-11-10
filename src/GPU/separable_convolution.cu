#include "separable_convolution.cuh"
#include "error.cuh"
#include <assert.h>


namespace gpu
{
    __constant__ double c_Gauss[KERNEL_LENGTH];
    __constant__ double c_GaussDeriv[KERNEL_LENGTH];

    void setGaussianKernel()
    {
        cudaMemcpyToSymbol(c_Gauss, h_Gauss, KERNEL_LENGTH * sizeof(double));
    }
    void setDerivGaussianKernel()
    {
        cudaMemcpyToSymbol(c_GaussDeriv, h_GaussDeriv, KERNEL_LENGTH * sizeof(double));
    }

#define ROWS_BLOCKDIM_X 64
#define ROWS_BLOCKDIM_Y 8
#define ROWS_RESULT_STEPS 8
#define ROWS_HALO_STEPS 1

    __global__ void convolutionRowsKernel(double *d_Dst, size_t dstPitch, const double *d_Src, size_t srcPitch,
                                          int imageW, int imageH, KernelType kernelType)
    {
        __shared__ double s_Data[ROWS_BLOCKDIM_Y]
                               [(ROWS_RESULT_STEPS + 2 * ROWS_HALO_STEPS) * ROWS_BLOCKDIM_X];

        // Offset to the left halo edge
        const int baseX =
            (blockIdx.x * ROWS_RESULT_STEPS - ROWS_HALO_STEPS) * blockDim.x + threadIdx.x;
        const int baseY = blockIdx.y * blockDim.y + threadIdx.y;

        if (baseX >= imageW || baseY >= imageH)
            return;

        const auto *src_line = (double *)((char *)d_Src + baseY * srcPitch);
        auto *dst_line = (double *)((char *)d_Dst + baseY * dstPitch);

        // Load main data
#pragma unroll
        for (int i = 0; i < 2 * ROWS_HALO_STEPS + ROWS_RESULT_STEPS; ++i)
        {
            const int idx = baseX + i * blockDim.x;
            s_Data[threadIdx.y][threadIdx.x + i * blockDim.x] =
                (idx < 0 || idx >= imageW) ? 0 : src_line[idx];
        }

        // Compute and store results
        __syncthreads();

#pragma unroll
        for (int i = ROWS_HALO_STEPS; i < ROWS_HALO_STEPS + ROWS_RESULT_STEPS; ++i)
        {
            const int idx = baseX + i * blockDim.x;
            if (idx < 0 || idx >= imageW)
                continue;

            double sum = 0;
            const double *kernel = kernelType == KernelType::GAUSSIAN ? c_Gauss : c_GaussDeriv;

#pragma unroll
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
                sum += kernel[KERNEL_RADIUS - j] * s_Data[threadIdx.y][threadIdx.x + i * blockDim.x + j];

            dst_line[idx] = sum;
        }
    }

    void convolutionRowsGPU(double *d_Dst, size_t dstPitch, const double *d_Src, size_t srcPitch,
                                       int imageW, int imageH, KernelType kernelType)
    {
        assert(ROWS_BLOCKDIM_X * ROWS_HALO_STEPS >= KERNEL_RADIUS);
        assert(imageW % (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X) == 0);
        assert(imageH % ROWS_BLOCKDIM_Y == 0);

        dim3 blocks(imageW / (ROWS_RESULT_STEPS * ROWS_BLOCKDIM_X), imageH / ROWS_BLOCKDIM_Y);
        dim3 threads(ROWS_BLOCKDIM_X, ROWS_BLOCKDIM_Y);

        convolutionRowsKernel<<<blocks, threads>>>(d_Dst, dstPitch, d_Src, srcPitch,
                                                                   imageW, imageH, kernelType);
        cudaDeviceSynchronize();
        abortError("convolutionRowsKernel() execution failed\n");
    }

#define COLUMNS_BLOCKDIM_X 8
#define COLUMNS_BLOCKDIM_Y 64
#define COLUMNS_RESULT_STEPS 8
#define COLUMNS_HALO_STEPS 1

    __global__ void convolutionColumnsKernel(double *d_Dst, size_t dstPitch, const double *d_Src, size_t srcPitch,
                                             int imageW, int imageH, KernelType kernelType)
    {
        __shared__ double
            s_Data[COLUMNS_BLOCKDIM_X]
                  [(COLUMNS_RESULT_STEPS + 2 * COLUMNS_HALO_STEPS) * COLUMNS_BLOCKDIM_Y + 1];

        // Offset to the upper halo edge
        const int baseX = blockIdx.x * blockDim.x + threadIdx.x;
        const int baseY =
            (blockIdx.y * COLUMNS_RESULT_STEPS - COLUMNS_HALO_STEPS) * blockDim.y + threadIdx.y;

        if (baseX >= imageW || baseY >= imageH)
            return;

#pragma unroll
        for (int i = 0; i < 2 * COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
        {
            const int idx = baseY + i * blockDim.y;
            s_Data[threadIdx.x][threadIdx.y + i * blockDim.y] =
                (idx < 0 || idx >= imageH) ? 0 :
                                           *((double *)((char *)d_Src + idx * srcPitch) + baseX);
        }
        // Compute and store results
        __syncthreads();

#pragma unroll
        for (int i = COLUMNS_HALO_STEPS; i < COLUMNS_HALO_STEPS + COLUMNS_RESULT_STEPS; i++)
        {
            const int idx = baseY + i * blockDim.y;
            if (idx < 0 || idx >= imageH)
                continue;

            double sum = 0;
            const double *kernel = kernelType == KernelType::GAUSSIAN ? c_Gauss : c_GaussDeriv;

#pragma unroll
            for (int j = -KERNEL_RADIUS; j <= KERNEL_RADIUS; j++)
            {
                sum += kernel[KERNEL_RADIUS - j]
                    * s_Data[threadIdx.x][threadIdx.y + i * blockDim.y + j];
            }

            auto dst_line = (double *)((char *)d_Dst + idx * dstPitch);
            dst_line[baseX] = sum;
        }
    }

    void convolutionColumnsGPU(double *d_Dst, size_t dstPitch, const double *d_Src, size_t srcPitch,
                                          int imageW, int imageH, KernelType kernelType)
    {
        assert(COLUMNS_BLOCKDIM_Y * COLUMNS_HALO_STEPS >= KERNEL_RADIUS);
        assert(imageW % COLUMNS_BLOCKDIM_X == 0);
        assert(imageH % (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y) == 0);

        dim3 blocks(imageW / COLUMNS_BLOCKDIM_X,
                    imageH / (COLUMNS_RESULT_STEPS * COLUMNS_BLOCKDIM_Y));
        dim3 threads(COLUMNS_BLOCKDIM_X, COLUMNS_BLOCKDIM_Y);

        convolutionColumnsKernel<<<blocks, threads>>>(d_Dst, dstPitch, d_Src, srcPitch,
                                                                      imageW, imageH, kernelType);
        cudaDeviceSynchronize();
        abortError("convolutionColumnsKernel() execution failed\n");
    }
}