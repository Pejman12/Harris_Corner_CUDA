//
// Created by pejman on 08/11/22.
//

#include "error.cuh"

namespace gpu
{

    [[gnu::noinline]] void _abortError(const char *msg, const char *fname, int line)
    {
        cudaError_t err = cudaGetLastError();
        if (err == cudaError_t::cudaSuccess)
            return;
        spdlog::error("{} ({}, line: {})", msg, fname, line);
        spdlog::error("Error {}: {}", cudaGetErrorName(err), cudaGetErrorString(err));
        std::exit(1);
    }

}
