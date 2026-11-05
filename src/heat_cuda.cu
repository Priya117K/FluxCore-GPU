#include <iostream>
#include <cuda_runtime.h>
#include <chrono>

#define N 512       // Grid dimension (512x512)
#define STEPS 1000  // Number of time steps

__global__ void heatKernel(const float* __restrict__ in_grid, float* __restrict__ out_grid, int n) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > 0 && row < n - 1 && col > 0 && col < n - 1) {
        float current = in_grid[row * n + col];
        float north   = in_grid[(row - 1) * n + col];
        float south   = in_grid[(row + 1) * n + col];
        float east    = in_grid[row * n + (col + 1)];
        float west    = in_grid[row * n + (col - 1)];

        out_grid[row * n + col] = current + 0.25f * (north + south + east + west - 4.0f * current);
    } else if (row < n && col < n) {
        out_grid[row * n + col] = in_grid[row * n + col];
    }
}

int main() {
    size_t size = N * N * sizeof(float);
    float *h_grid = new float[N * N];

    for (int i = 0; i < N * N; i++) { h_grid[i] = 0.0f; }
    h_grid[(N / 2) * N + (N / 2)] = 1000.0f;

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_grid, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x, 
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < STEPS; ++step) {
        heatKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, N);
        float *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end_time - start_time;

    std::cout << "CUDA Kernel execution time: " << elapsed.count() << " ms\n";

    cudaMemcpy(h_grid, d_in, size, cudaMemcpyDeviceToHost);
    delete[] h_grid;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
}
