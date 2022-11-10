//
// Created by pejman on 08/11/22.
//

#ifndef HARRIS_ERROR_CUH
#define HARRIS_ERROR_CUH

#include <spdlog/spdlog.h>

namespace gpu
{

    [[gnu::noinline]] void _abortError(const char *msg, const char *fname, int line);

#define abortError(msg) _abortError(msg, __FUNCTION__, __LINE__)

}

#endif // HARRIS_ERROR_CUH
