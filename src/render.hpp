#pragma once
#include <memory>
#include <vector>
#include <cmath>

#define EPS 1
#define MAX_KPTS 2000
#define KERNEL_RADIUS 3
#define KERNEL_LENGTH 7
#define THRESHOLD 0.2
#define DISTANCE 15

struct point {
    int x;
    int y;
    double score;
};

//#include "CPU/corners_cpu.hpp"
//#include "GPU/corners_gpu.cuh"

