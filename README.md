# FluxCore-GPU

A high-performance computing (HPC) project comparing sequential CPU performance against parallelized CUDA implementations for solving the 2D heat diffusion equation.

## Project Structure
- `src/heat_cpu.cpp`: Sequential C++ baseline implementation.
- `src/heat_cuda.cu`: GPU-accelerated CUDA implementation (coming soon).

## Performance Benchmark
- **Grid Size:** 512 x 512
- **Iterations:** 1000 steps
- **CPU Baseline Execution Time:** (Measured via `std::chrono`)
