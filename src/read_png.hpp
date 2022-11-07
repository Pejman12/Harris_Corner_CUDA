#pragma once
#include <spdlog/spdlog.h>
#include <memory>
#include <png.h>

png_bytep* read_png(const char* filename, int *width, int *height);
