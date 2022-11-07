# Harris corner detection with cuda kernels

## Build and run

```bash
mkdir build
cd build
cmake ..
make render -j <nb_cores>
./render [-i <input_image>] [-o <output_image>] [-m <mode 'GPU/CPU'>]
```

## Benchmark

```bash
make bench -j <nb_cores>  # building could fail because of google benchmark library 
                          # you may need to add a missing import in the built files
./bench
```
