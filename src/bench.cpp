#include <benchmark/benchmark.h>
#include "read_png.hpp"
#include "render.hpp"

constexpr int niteration = 30;
int width;
int height;
const auto image = read_png("windows-logo-digital-art-8k-pv.png", &width, &height);

void BM_Rendering_cpu(benchmark::State& st)
{
    int ncorners = 0;
    for (auto _ : st)
        cpu::find_corners_cpu(image, width, height, &ncorners);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu1(benchmark::State& st)
{
    int ncorners = 0;
    for (auto _ : st)
        gpu::find_corners_gpu_1st_implem(image, width, height, &ncorners);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

void BM_Rendering_gpu2(benchmark::State& st)
{
    int ncorners = 0;
    for (auto _ : st)
        gpu::find_corners_gpu(image, width, height, &ncorners);

    st.counters["frame_rate"] = benchmark::Counter(st.iterations(), benchmark::Counter::kIsRate);
}

BENCHMARK(BM_Rendering_cpu)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Iterations(3);

BENCHMARK(BM_Rendering_gpu1)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Iterations(niteration);

BENCHMARK(BM_Rendering_gpu2)
->Unit(benchmark::kMillisecond)
->UseRealTime()
->Iterations(niteration);

BENCHMARK_MAIN();
