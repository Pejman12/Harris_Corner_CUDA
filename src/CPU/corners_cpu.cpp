#include <omp.h>
#include "corners_cpu.hpp"

namespace cpu {

    static void delete_matrix(matrix_t matrix, int n) {
#pragma omp parallel for schedule(dynamic) shared(matrix, n) default(none)
        for (int i = 0; i < n; ++i)
            free(matrix[i]);
        free(matrix);
    }

    static double image_min(const matrix_t buffer, const int width, const int height) {
        double min = std::numeric_limits<double>::max();
#pragma omp parallel for schedule(dynamic) shared(buffer, height, width) reduction(min:min) default(none)
        for (auto i = 0; i < height; ++i)
            for (auto j = 0; j < width; ++j)
                min = std::fmin(min, buffer[i][j]);
        return min;
    }

    static double image_max(const matrix_t buffer, const int width, const int height) {
        double max = std::numeric_limits<double>::min();
#pragma omp parallel for schedule(dynamic) shared(buffer, height, width) reduction(max:max) default(none)
        for (auto i = 0; i < height; ++i)
            for (auto j = 0; j < width; ++j)
                max = std::fmax(max, buffer[i][j]);
        return max;
    }

    static matrix_t convolve(const matrix_t buffer, const int width, const int height, const double kernel[7][7]) {
        auto result = (matrix_t) malloc(height * sizeof(double *));
        for (auto i = 0; i < height; result[i++] = (double *) malloc(width * sizeof(double)));

#pragma omp parallel for schedule(dynamic) shared(buffer, result, height, width, kernel) default(none)
        for (auto i = 0; i < height; ++i)
            for (auto j = 0; j < width; ++j) {
                double sum = 0;
                for (auto k = -KERNEL_RADIUS; k <= KERNEL_RADIUS; ++k)
                    for (auto l = -KERNEL_RADIUS; l <= KERNEL_RADIUS; ++l) {
                        if (i + k >= 0 && i + k < height && j + l >= 0 && j + l < width)
                            sum += buffer[i + k][j + l] * kernel[k + KERNEL_RADIUS][l + KERNEL_RADIUS];
                    }
                result[i][j] = sum;
            }

        return result;
    }

    static matrix_t matmult(const matrix_t a, const matrix_t b, const int width, const int height) {
        auto res = (matrix_t) malloc(height * sizeof(unsigned char *));
        for (auto i = 0; i < height; res[i++] = (double *) malloc(width * sizeof(double)));

#pragma omp parallel for schedule(dynamic) shared(a, b, res, height, width) default(none)
        for (auto i = 0; i < height; ++i)
            for (auto j = 0; j < width; ++j)
                res[i][j] = a[i][j] * b[i][j];
        return res;
    }

    static matrix_t compute_harris_response(const matrix_t buffer, const int width, const int height) {
        auto res = (matrix_t) malloc(height * sizeof(double *));
        for (auto i = 0; i < height; res[i++] = (double *) malloc(width * sizeof(double)));

        auto Ix = convolve(buffer, width, height, DerivGaussX);
        auto Iy = convolve(buffer, width, height, DerivGaussY);

        auto Ix2 = matmult(Ix, Ix, width, height);
        auto Iy2 = matmult(Iy, Iy, width, height);
        auto IxIy = matmult(Ix, Iy, width, height);
        delete_matrix(Ix, height);
        delete_matrix(Iy, height);

        auto Wx2 = convolve(Ix2, width, height, Gauss);
        auto Wy2 = convolve(Iy2, width, height, Gauss);
        auto Wxy = convolve(IxIy, width, height, Gauss);

        delete_matrix(Ix2, height);
        delete_matrix(Iy2, height);
        delete_matrix(IxIy, height);

#pragma omp parallel for schedule(dynamic) shared(Wx2, Wy2, Wxy, res, height, width) default(none)
        for (auto i = 0; i < height; ++i)
            for (auto j = 0; j < width; ++j)
                res[i][j] = (Wx2[i][j] * Wy2[i][j] - Wxy[i][j] * Wxy[i][j]) / (Wx2[i][j] + Wy2[i][j] + EPS);

        delete_matrix(Wx2, height);
        delete_matrix(Wy2, height);
        delete_matrix(Wxy, height);
        return res;
    }

    static matrix_t dilate(const matrix_t buffer, const int width, const int height, const int size) {
        auto res = (matrix_t) malloc(height * sizeof(double *));
        for (auto i = 0; i < height; res[i++] = (double *) malloc(width * sizeof(double)));

#pragma omp parallel for schedule(dynamic) shared(buffer, res, height, width, size) default(none)
        for (auto i = 0; i < height; ++i)
            for (auto j = 0; j < width; ++j) {
                auto max = std::numeric_limits<double>::min();
                for (auto k = -size; k <= size; ++k)
                    for (auto l = -size; l <= size; ++l)
                        if (i + k >= 0 && i + k < height && j + l >= 0 && j + l < width)
                            max = std::fmax(max, buffer[i + k][j + l]);
                res[i][j] = max;
            }
        return res;
    }

    static point *insert_kpts(point *kpts, const matrix_t harris_resp, int *nb_kpts,
                              int max_kpts, const int i, const int j) {
        if (*nb_kpts < max_kpts)
            kpts = (point *) realloc(kpts, ++(*nb_kpts) * sizeof(point));
        const auto test = harris_resp[i][j];
        int idx = 0;
        while (idx < *nb_kpts - 1 && harris_resp[kpts[idx].y][kpts[idx].x] >= test)
            ++idx;
        for (auto k = *nb_kpts - 1; k > idx; --k) {
            kpts[k].x = kpts[k - 1].x;
            kpts[k].y = kpts[k - 1].y;
        }
        kpts[idx].x = j;
        kpts[idx].y = i;
        return kpts;
    }

    static point *find_corners(const matrix_t buffer, const int width, const int height, int *nb_kpts,
                               const int max_kpts, const double threshold, const int distance) {
        const auto harris_resp = compute_harris_response(buffer, width, height);
        const auto harris_min = image_min(harris_resp, width, height);
        const auto harris_max = image_max(harris_resp, width, height);
        const auto compute_threshold = harris_min + (harris_max - harris_min) * threshold;
        auto mask = dilate(harris_resp, width, height, distance);

#pragma omp parallel for schedule(dynamic) shared(mask, harris_resp, compute_threshold, height, width) default(none)
        for (auto i = 0; i < height; ++i)
            for (auto j = 0; j < width; ++j)
                mask[i][j] = (harris_resp[i][j] > compute_threshold) & (mask[i][j] == harris_resp[i][j]);

        *nb_kpts = 0;
        point *kpts = nullptr;

#pragma omp parallel for schedule(dynamic) shared(mask, harris_resp, height, width, nb_kpts, kpts, max_kpts) default(none)
        for (auto i = 0; i < height; ++i)
            for (auto j = 0; j < width; ++j)
                if (mask[i][j] != 0)
#pragma omp critical
                    kpts = insert_kpts(kpts, harris_resp, nb_kpts, max_kpts, i, j);

        delete_matrix(harris_resp, height);
        delete_matrix(mask, height);

        return kpts;
    }

    static matrix_t to_matrix(unsigned char **buffer, const int width, const int height) {
        auto res = (matrix_t) malloc(height * sizeof(double *));
        for (auto i = 0; i < height; res[i++] = (double *) malloc(width * sizeof(double)));

#pragma omp parallel for schedule(dynamic) shared(buffer, res, height, width) default(none)
        for (auto i = 0; i < height; ++i)
            for (auto j = 0; j < width; ++j)
                res[i][j] = buffer[i][j];

        return res;
    }

    point *find_corners_cpu(unsigned char **buffer, const int width, const int height,
                            int *nb_kpts) {
        auto buffer_matrix = to_matrix(buffer, width, height);
        auto res = find_corners(buffer_matrix, width, height, nb_kpts, MAX_KPTS,
                                THRESHOLD, DISTANCE);
        delete_matrix(buffer_matrix, height);
        return res;
    }
}