#include <ostream>
#include <CLI/CLI.hpp>
#include "render.hpp"
#include "read_png.hpp"

// write the point found by the corner detector in the outputfile
void write_points(const char* filename, point *corners, int nb)
{
    std::ofstream file(filename);

    for (auto i = 0; i < nb; ++i)
        file << corners[i].x << " " << corners[i].y << "\n";

    file.close();
}

int main(int argc, char** argv)
{
    (void) argc;
    (void) argv;

    std::string input_file = "input.png";
    std::string output_file = "output.txt";
    std::string mode = "CPU";

    CLI::App app{"mandel"};
    app.add_option("-i", input_file, "Input image");
    app.add_option("-o", output_file, "Output file with corner coordinates");
    app.add_set("-m", mode, {"GPU", "CPU"}, "Either 'GPU' or 'CPU'");

    CLI11_PARSE(app, argc, argv);

    int width, height;
    auto image = read_png(input_file.c_str(), &width, &height);
    // Rendering
    spdlog::info("Running on ", input_file);
    if (mode == "CPU")
    {
        int nb_kpts = 0;
        auto corners = cpu::find_corners_cpu(image, width, height, &nb_kpts);
        std::cout << "Found " << nb_kpts << " corners" << std::endl;
        write_points(output_file.c_str(), corners, nb_kpts);
        free(corners);
    }

    else if (mode == "GPU")
    {
        int nb_kpts = 0;
        auto corners = gpu::find_corners_gpu(image, width, height, &nb_kpts);
        std::cout << "Found " << nb_kpts << " corners" << std::endl;
        write_points(output_file.c_str(), corners, nb_kpts);
        free(corners);
    }

    for (auto i = 0; i < height; free(image[i++]));
    free(image);

    // Save
    spdlog::info("Done processing");
}
