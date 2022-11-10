#include <benchmark/benchmark.h>
#include "read_png.hpp"
#include "GPU/corners_gpu.cuh"
#include "CPU/corners_cpu.hpp"

constexpr int niteration = 30;
int width;
int height;
const auto image = read_png("imgs/hamburg_fHD.png", &width, &height);

void BM_Rendering_cpu(benchmark::State& st)
{
    int ncorners = 0;
    for (auto _ : st)
        cpu::find_corners_cpu(image, width, height, &ncorners);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu(benchmark::State& st)
{
    int ncorners = 0;
    for (auto _ : st)
        gpu::find_corners_gpu(image, width, height, &ncorners);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

//BENCHMARK(BM_Rendering_cpu)
//->Unit(benchmark::kMillisecond)
//->UseRealTime()
//->Iterations(3);

BENCHMARK(BM_Rendering_gpu)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Iterations(niteration);

BENCHMARK_MAIN();
