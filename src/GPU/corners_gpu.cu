#include "corners_gpu.cuh"
#include <spdlog/spdlog.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>
#include <thrust/remove.h>

namespace gpu
{

    [[gnu::noinline]] void _abortError(const char *msg, const char *fname, int line)
    {
        cudaError_t err = cudaGetLastError();
        spdlog::error("{} ({}, line: {})", msg, fname, line);
        spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
        std::exit(1);
    }

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

    __global__ void convolve(double *buffer, const int width, const int height, size_t bufferPitch,
                             double *kernel, size_t kernelPitch, double *res, size_t resPitch)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        auto *res_line = (double *)((char *)res + y * resPitch);
        double sum = 0;
        for (auto i = -KERNEL_SIZE; i <= KERNEL_SIZE; ++i)
        {
            if (y + i < 0 || y + i >= height)
                continue;
            const auto *buffer_line = (double *)((char *)buffer + (y + i) * bufferPitch);
            const auto *kernel_line = (double *)((char *)kernel + (i + KERNEL_SIZE) * kernelPitch);
            for (auto j = -KERNEL_SIZE; j <= KERNEL_SIZE; ++j)
            {
                if (x + j < 0 || x + j >= width)
                    continue;
                sum += buffer_line[x + j] * kernel_line[j + KERNEL_SIZE];
            }
        }
        res_line[x] = sum;
    }

    __global__ void matmult(double *a, size_t aPitch, double *b, size_t bPitch, const int width,
                            const int height, double *res, size_t resPitch)
    {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        const auto y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        const auto *a_line = (double *)((char *)a + y * aPitch);
        const auto *b_line = (double *)((char *)b + y * bPitch);
        auto *res_line = (double *)((char *)res + y * resPitch);
        res_line[x] = a_line[x] * b_line[x];
    }

    __global__ void compute_harris(double *Wx2, size_t Wx2Pitch, double *Wy2, size_t Wy2Pitch,
                                   double *Wxy, size_t WxyPitch, const int width, const int height,
                                   double *res, size_t resPitch)
    {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        const auto y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        const auto *Wx2_line = (double *)((char *)Wx2 + y * Wx2Pitch);
        const auto *Wy2_line = (double *)((char *)Wy2 + y * Wy2Pitch);
        const auto *Wxy_line = (double *)((char *)Wxy + y * WxyPitch);
        auto *res_line = (double *)((char *)res + y * resPitch);
        double det = Wx2_line[x] * Wy2_line[x] - Wxy_line[x] * Wxy_line[x];
        double trace = Wx2_line[x] + Wy2_line[x];
        res_line[x] = det / (trace + EPS);
    }

    __global__ void dilate(double *buffer, size_t bufferPitch, const int width, const int height,
                           const int size, double *res, size_t resPitch, double minDoubleValue)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        auto *res_line = (double *)((char *)res + y * resPitch);
        auto max = minDoubleValue;
        for (auto i = -size; i <= size; ++i)
        {
            if (y + i < 0 || y + i >= height)
                continue;
            const auto *buffer_line = (double *)((char *)buffer + (y + i) * bufferPitch);
            for (auto j = -size; j <= size; ++j)
            {
                if (x + j < 0 || x + j >= width)
                    continue;
                if (buffer_line[x + j] > max)
                    max = buffer_line[x + j];
            }
        }
        res_line[x] = max;
    }

    template <unsigned int radio>
    __global__ void DilationSharedStep2(double *src, size_t spitch, double *dst, size_t dpitch,
                                        int width, int height, int tile_w, int tile_h)
    {
        extern __shared__ double smem[];
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int x = bx * tile_w + tx;
        int y = by * tile_h + ty - radio;
        // auto smem_line = (double *)((char *)smem + ty * blockDim.x);
        smem[ty * blockDim.x + tx] = 0.f;
        __syncthreads();
        if (x >= width || y < 0 || y >= height)
        {
            return;
        }
        const auto src_line = (double *)((char *)src + y * spitch);
        smem[ty * blockDim.x + tx] = src_line[x];
        __syncthreads();
        if (y < (by * tile_h) || y >= ((by + 1) * tile_h))
        {
            return;
        }
        double *smem_thread = &smem[(ty - radio) * blockDim.x + tx];
        double val = smem_thread[0];
        for (int yy = 1; yy <= 2 * radio; yy++)
        {
            val = fmax(val, smem_thread[yy * blockDim.x]);
        }
        auto dst_line = (double *)((char *)dst + y * dpitch);
        dst_line[x] = val;
    }

    template <unsigned int radio>
    __global__ void DilationSharedStep1(double *src, size_t spitch, double *dst, size_t dpitch,
                                        int width, int height, int tile_w, int tile_h)
    {
        extern __shared__ double smem[];
        int tx = threadIdx.x;
        int ty = threadIdx.y;
        int bx = blockIdx.x;
        int by = blockIdx.y;
        int x = bx * tile_w + tx - radio;
        int y = by * tile_h + ty;
        // auto smem_line = (double *)(smem + ty * blockDim.x);
        smem[ty * blockDim.x + tx] = 0.f;
        __syncthreads();
        if (x < 0 || x >= width || y >= height)
        {
            return;
        }
        const auto src_line = (double *)((char *)src + y * spitch);
        smem[ty * blockDim.x + tx] = src_line[x];
        __syncthreads();
        if (x < (bx * tile_w) || x >= ((bx + 1) * tile_w))
        {
            return;
        }
        double *smem_thread = &smem[ty * blockDim.x + tx - radio];
        double val = smem_thread[0];
        for (int xx = 1; xx <= 2 * radio; xx++)
        {
            val = fmax(val, smem_thread[xx]);
        }
        auto dst_line = (double *)((char *)dst + y * dpitch);
        dst_line[x] = val;
    }

    template <unsigned int radio>
    void DilationTwoStepsShared(double *src, size_t spitch, double *dst, size_t dpitch, int width,
                                int height)
    {
        double *temp;
        size_t tempPitch;
        if (cudaMallocPitch(&temp, &tempPitch, width * sizeof(double), height) != cudaSuccess)
            abortError("Fail buffer allocation");
        int tile_w = 640;
        int tile_h = 1;
        dim3 block2(tile_w + (2 * radio), tile_h);
        dim3 grid2(ceil((float)width / tile_w), ceil((float)height / tile_h));
        DilationSharedStep1<radio><<<grid2, block2, block2.y * block2.x * sizeof(double)>>>(
            src, spitch, temp, tempPitch, width, height, tile_w, tile_h);
        cudaDeviceSynchronize();
        tile_w = 8;
        tile_h = 64;
        dim3 block3(tile_w, tile_h + (2 * radio));
        dim3 grid3(ceil((float)width / tile_w), ceil((float)height / tile_h));
        DilationSharedStep2<radio><<<grid3, block3, block3.y * block3.x * sizeof(double)>>>(
            temp, tempPitch, dst, dpitch, width, height, tile_w, tile_h);
        cudaDeviceSynchronize();
        cudaFree(temp);
    }

    __global__ void harris_mask(double *harris, size_t harrisPitch, double *dilate,
                                size_t dilatePitch, const int width, const int height, bool *mask,
                                size_t maskPitch, const double threshold)
    {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        const auto y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        const auto *harris_line = (double *)((char *)harris + y * harrisPitch);
        const auto *dilate_line = (double *)((char *)dilate + y * dilatePitch);
        auto *mask_line = (bool *)((char *)mask + y * maskPitch);
        mask_line[x] = (harris_line[x] > threshold) & (dilate_line[x] == harris_line[x]);
    }

    __global__ void apply_mask_harris(double *harris, size_t harrisPitch, bool *mask, size_t maskPitch,
                                      const int width, const int height)
    {
        const auto x = blockDim.x * blockIdx.x + threadIdx.x;
        const auto y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        auto *harris_line = (double *)((char *)harris + y * harrisPitch);
        const auto *mask_line = (bool *)((char *)mask + y * maskPitch);
        harris_line[x] = mask_line[x] ? harris_line[x] : 0;
    }

    __global__ void harris_to_kpts(double *harris, size_t harrisPitch, point *kpts, size_t kptsPitch,
                                   const int width, const int height)
    {
        const int x = blockDim.x * blockIdx.x + threadIdx.x;
        const int y = blockDim.y * blockIdx.y + threadIdx.y;

        if (x >= width || y >= height)
            return;

        const auto *harris_line = (double *)((char *)harris + y * harrisPitch);
        auto *kpts_line = (point *)((char *)kpts + y * kptsPitch);
        kpts_line[x].x = x;
        kpts_line[x].y = y;
        kpts_line[x].score = harris_line[x];
    }

    struct remove_null_kpts
    {
        __host__ __device__
            bool operator()(const point &kpt) const
        {
            return kpt.score == 0;
        }
    };

    struct sort_kpts
    {
        __host__ __device__
            bool operator()(const point &a, const point &b) const
        {
            return a.score < b.score;
        }
    };

    static double *to_matrix(unsigned char **buffer, const int width, const int height)
    {
        auto res = (double *)malloc(height * width * sizeof(double));

#pragma omp parallel for schedule(dynamic) shared(buffer, res, height, width) default(none)
        for (auto i = 0; i < height; ++i)
            for (auto j = 0; j < width; ++j)
                res[i * width + j] = buffer[i][j];

        return res;
    }

    point *find_corners_gpu(unsigned char **buffer_, const int width, const int height,
                            int *nb_kpts)
    {
        int bsize = 32;
        int w = std::ceil((float)width / bsize);
        int h = std::ceil((float)height / bsize);

        spdlog::debug("running kernel of size ({},{})", w, h);

        dim3 dimGrid(w, h);
        dim3 dimBlock(bsize, bsize);

        // Device image buffer
        double *devBuffer;
        size_t BufferPitch;
        if (cudaMallocPitch(&devBuffer, &BufferPitch, width * sizeof(double), height)
            != cudaSuccess)
            abortError("Fail buffer allocation");
        const auto buffer = to_matrix(buffer_, width, height);
        if (cudaMemcpy2D(devBuffer, BufferPitch, buffer, width * sizeof(double),
                         width * sizeof(double), height, cudaMemcpyHostToDevice)
            != cudaSuccess)
            abortError("Fail buffer copy");
        free(buffer);

        // Device Gaussian kernel
        double *devGauss;
        size_t GaussPitch;
        if (cudaMallocPitch(&devGauss, &GaussPitch, KERNEL_LENGTH * sizeof(double), KERNEL_LENGTH)
            != cudaSuccess)
            abortError("Fail Gaussian kernel allocation");
        if (cudaMemcpy2D(devGauss, GaussPitch, Gauss, KERNEL_LENGTH * sizeof(double),
                         KERNEL_LENGTH * sizeof(double), KERNEL_LENGTH, cudaMemcpyHostToDevice)
            != cudaSuccess)
            abortError("Fail Gaussian kernel copy");

        // Device DerivGaussX kernel
        double *devDerivGaussX;
        size_t DerivGaussXPitch;
        if (cudaMallocPitch(&devDerivGaussX, &DerivGaussXPitch, KERNEL_LENGTH * sizeof(double),
                            KERNEL_LENGTH)
            != cudaSuccess)
            abortError("Fail DerivGaussX kernel allocation");
        if (cudaMemcpy2D(devDerivGaussX, DerivGaussXPitch, DerivGaussX,
                         KERNEL_LENGTH * sizeof(double), KERNEL_LENGTH * sizeof(double),
                         KERNEL_LENGTH, cudaMemcpyHostToDevice)
            != cudaSuccess)
            abortError("Fail DerivGaussX kernel copy");

        // Device DerivGaussY kernel
        double *devDerivGaussY;
        size_t DerivGaussYPitch;
        if (cudaMallocPitch(&devDerivGaussY, &DerivGaussYPitch, KERNEL_LENGTH * sizeof(double),
                            KERNEL_LENGTH)
            != cudaSuccess)
            abortError("Fail DerivGaussY kernel allocation");
        if (cudaMemcpy2D(devDerivGaussY, DerivGaussYPitch, DerivGaussY,
                         KERNEL_LENGTH * sizeof(double), KERNEL_LENGTH * sizeof(double),
                         KERNEL_LENGTH, cudaMemcpyHostToDevice)
            != cudaSuccess)
            abortError("Fail DerivGaussY kernel copy");

        // Device Ix
        double *devIx;
        size_t IxPitch;
        if (cudaMallocPitch(&devIx, &IxPitch, width * sizeof(double), height) != cudaSuccess)
            abortError("Fail Ix allocation");
        convolve<<<dimGrid, dimBlock>>>(devBuffer, width, height, BufferPitch, devDerivGaussX,
                                        DerivGaussXPitch, devIx, IxPitch);
        cudaDeviceSynchronize();
        cudaFree(devDerivGaussX);

        // Device Iy
        double *devIy;
        size_t IyPitch;
        if (cudaMallocPitch(&devIy, &IyPitch, width * sizeof(double), height) != cudaSuccess)
            abortError("Fail Iy allocation");
        convolve<<<dimGrid, dimBlock>>>(devBuffer, width, height, BufferPitch, devDerivGaussY,
                                        DerivGaussYPitch, devIy, IyPitch);
        cudaDeviceSynchronize();
        cudaFree(devDerivGaussY);
        cudaFree(devBuffer);

        // Device Ix2
        double *devIx2;
        size_t Ix2Pitch;
        if (cudaMallocPitch(&devIx2, &Ix2Pitch, width * sizeof(double), height) != cudaSuccess)
            abortError("Fail Ix2 allocation");
        matmult<<<dimGrid, dimBlock>>>(devIx, IxPitch, devIx, IxPitch, width, height, devIx2,
                                       Ix2Pitch);
        cudaDeviceSynchronize();

        // Device Iy2
        double *devIy2;
        size_t Iy2Pitch;
        if (cudaMallocPitch(&devIy2, &Iy2Pitch, width * sizeof(double), height) != cudaSuccess)
            abortError("Fail Iy2 allocation");
        matmult<<<dimGrid, dimBlock>>>(devIy, IyPitch, devIy, IyPitch, width, height, devIy2,
                                       Iy2Pitch);
        cudaDeviceSynchronize();

        // Device Ixy
        double *devIxy;
        size_t IxyPitch;
        if (cudaMallocPitch(&devIxy, &IxyPitch, width * sizeof(double), height) != cudaSuccess)
            abortError("Fail Ixy allocation");
        matmult<<<dimGrid, dimBlock>>>(devIx, IxPitch, devIy, IyPitch, width, height, devIxy,
                                       IxyPitch);
        cudaDeviceSynchronize();
        cudaFree(devIx);
        cudaFree(devIy);

        // Device Wx2
        double *devWx2;
        size_t Wx2Pitch;
        if (cudaMallocPitch(&devWx2, &Wx2Pitch, width * sizeof(double), height) != cudaSuccess)
            abortError("Fail Wx2 allocation");
        convolve<<<dimGrid, dimBlock>>>(devIx2, width, height, Ix2Pitch, devGauss, GaussPitch,
                                        devWx2, Wx2Pitch);
        cudaDeviceSynchronize();

        // Device Wy2
        double *devWy2;
        size_t Wy2Pitch;
        if (cudaMallocPitch(&devWy2, &Wy2Pitch, width * sizeof(double), height) != cudaSuccess)
            abortError("Fail Wy2 allocation");
        convolve<<<dimGrid, dimBlock>>>(devIy2, width, height, Iy2Pitch, devGauss, GaussPitch,
                                        devWy2, Wy2Pitch);
        cudaDeviceSynchronize();

        // Device Wxy
        double *devWxy;
        size_t WxyPitch;
        if (cudaMallocPitch(&devWxy, &WxyPitch, width * sizeof(double), height) != cudaSuccess)
            abortError("Fail Wxy allocation");
        convolve<<<dimGrid, dimBlock>>>(devIxy, width, height, IxyPitch, devGauss, GaussPitch,
                                        devWxy, WxyPitch);
        cudaDeviceSynchronize();
        cudaFree(devIx2);
        cudaFree(devIy2);
        cudaFree(devIxy);
        cudaFree(devGauss);

        // Device Harris
        double *devHarris;
        size_t HarrisPitch;
        if (cudaMallocPitch(&devHarris, &HarrisPitch, width * sizeof(double), height)
            != cudaSuccess)
            abortError("Fail Harris allocation");
        compute_harris<<<dimGrid, dimBlock>>>(devWx2, Wx2Pitch, devWy2, Wy2Pitch, devWxy, WxyPitch,
                                              width, height, devHarris, HarrisPitch);
        cudaDeviceSynchronize();
        cudaFree(devWx2);
        cudaFree(devWy2);
        cudaFree(devWxy);

        thrust::device_vector<double> harris_vec(width * height);

        if (cudaMemcpy2D(thrust::raw_pointer_cast(harris_vec.data()), width * sizeof(double),
                         devHarris, HarrisPitch, width * sizeof(double), height,
                         cudaMemcpyDeviceToDevice)
            != cudaSuccess)
            abortError("Fail FlattenHarris copy");

        const auto harris_minmax = thrust::minmax_element(harris_vec.begin(), harris_vec.end());
        auto harris_min = *harris_minmax.first;
        auto harris_max = *harris_minmax.second;
        const auto compute_threshold = harris_min + (harris_max - harris_min) * THRESHOLD;

        // Device HarrisDilate
        double *devHarrisDilate;
        size_t HarrisDilatePitch;
        if (cudaMallocPitch(&devHarrisDilate, &HarrisDilatePitch, width * sizeof(double), height)
            != cudaSuccess)
            abortError("Fail HarrisDilate allocation");
        DilationTwoStepsShared<DISTANCE>(devHarris, HarrisPitch, devHarrisDilate, HarrisDilatePitch,
                                         width, height);

        // Device HarrisMask
        bool *devHarrisMask;
        size_t HarrisMaskPitch;
        if (cudaMallocPitch(&devHarrisMask, &HarrisMaskPitch, width * sizeof(bool), height)
            != cudaSuccess)
            abortError("Fail HarrisMask allocation");
        harris_mask<<<dimGrid, dimBlock>>>(devHarris, HarrisPitch, devHarrisDilate,
                                           HarrisDilatePitch, width, height, devHarrisMask,
                                           HarrisMaskPitch, compute_threshold);
        cudaDeviceSynchronize();
        cudaFree(devHarrisDilate);

        apply_mask_harris<<<dimGrid, dimBlock>>>(devHarris, HarrisPitch, devHarrisMask, HarrisMaskPitch, width, height);
        cudaDeviceSynchronize();
        cudaFree(devHarrisMask);

        // Device Kpts
        point *devKpts;
        size_t KptsPitch;
        if (cudaMallocPitch(&devKpts, &KptsPitch, width * sizeof(point), height) != cudaSuccess)
            abortError("Fail Kpts allocation");
        harris_to_kpts<<<dimGrid, dimBlock>>>(devHarris, HarrisPitch, devKpts, KptsPitch, width, height);
        cudaDeviceSynchronize();
        cudaFree(devHarris);

        thrust::device_vector<point> kpts_vec(width * height);

        if (cudaMemcpy2D(thrust::raw_pointer_cast(kpts_vec.data()), width * sizeof(point), devKpts, KptsPitch,
                         width * sizeof(point), height, cudaMemcpyDeviceToDevice) != cudaSuccess)
            abortError("Fail Kpts copy");
        cudaFree(devKpts);

        remove_null_kpts remove_struct;
        sort_kpts sort_struct;

        const auto new_last = thrust::remove_if(kpts_vec.begin(), kpts_vec.end(), remove_struct);

        thrust::sort(kpts_vec.begin(), new_last, sort_struct);

        *nb_kpts = new_last - kpts_vec.begin();
        *nb_kpts = std::min(MAX_KPTS, *nb_kpts);
        auto *kpts = (point *)malloc(*nb_kpts * sizeof(point));
        if (cudaMemcpy(kpts, thrust::raw_pointer_cast(kpts_vec.data()), *nb_kpts * sizeof(point), cudaMemcpyDeviceToHost) != cudaSuccess)
            abortError("Fail Kpts copy");

        return kpts;
    }
}