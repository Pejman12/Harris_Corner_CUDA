#ifndef GPGPU_CORNERS_HPP_GPU
#define GPGPU_CORNERS_HPP_GPU

#include "../render.hpp"

namespace gpu {

    const double Gauss[KERNEL_LENGTH * KERNEL_LENGTH] = {
            0.00012341, 0.00150344, 0.00673795, 0.011109, 0.00673795, 0.00150344, 0.00012341,
            0.00150344, 0.0183156, 0.082085, 0.135335, 0.082085, 0.0183156, 0.00150344,
            0.00673795, 0.082085, 0.367879, 0.606531, 0.367879, 0.082085, 0.00673795,
            0.011109, 0.135335, 0.606531, 1, 0.606531, 0.135335, 0.011109,
            0.00673795, 0.082085, 0.367879, 0.606531, 0.367879, 0.082085, 0.00673795,
            0.00150344, 0.0183156, 0.082085, 0.135335, 0.082085, 0.0183156, 0.00150344,
            0.00012341, 0.00150344, 0.00673795, 0.011109, 0.00673795, 0.00150344, 0.00012341
    };

    const double DerivGaussX[KERNEL_LENGTH * KERNEL_LENGTH] = {
            0.000370229, 0.00451032, 0.0202138, 0.033327, 0.0202138, 0.00451032, 0.000370229,
            0.00300688, 0.0366313, 0.16417, 0.270671, 0.16417, 0.0366313, 0.00300688,
            0.00673795, 0.082085, 0.367879, 0.606531, 0.367879, 0.082085, 0.00673795,
            0, 0, 0, 0, 0, 0, 0,
            -0.00673795, -0.082085, -0.367879, -0.606531, -0.367879, -0.082085, -0.00673795,
            -0.00300688, -0.0366313, -0.16417, -0.270671, -0.16417, -0.0366313, -0.00300688,
            -0.000370229, -0.00451032, -0.0202138, -0.033327, -0.0202138, -0.00451032, -0.000370229
    };

    const double DerivGaussY[KERNEL_LENGTH * KERNEL_LENGTH] = {
            0.000370229, 0.00300688, 0.00673795, 0, -0.00673795, -0.00300688, -0.000370229,
            0.00451032, 0.0366313, 0.082085, 0, -0.082085, -0.0366313, -0.00451032,
            0.0202138, 0.16417, 0.367879, 0, -0.367879, -0.16417, -0.0202138,
            0.033327, 0.270671, 0.606531, 0, -0.606531, -0.270671, -0.033327,
            0.0202138, 0.16417, 0.367879, 0, -0.367879, -0.16417, -0.0202138,
            0.00451032, 0.0366313, 0.082085, 0, -0.082085, -0.0366313, -0.00451032,
            0.000370229, 0.00300688, 0.00673795, 0, -0.00673795, -0.00300688, -0.000370229
    };


    extern "C" point *find_corners_gpu(unsigned char **buffer_, const int width, const int height, int *nb_kpts);

}

#endif //GPGPU_CORNERS_HPP_GPU
