# Harris corner detection with cuda kernels

## Build and run

```bash
mkdir build
cd build
cmake ..
make -j <nb_cores> render
./render [-i <input_image>] [-o <output_image>] [-m <mode 'GPU/CPU'>]
```

## Benchmark

```bash
make -j <nb_cores> bench  # building could fail because of google benchmark library 
                          # you may need to add a missing import in the built files
./bench
```
