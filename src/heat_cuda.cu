
// Host function to compute max temperature for metrics tracking
float findMaxTemperature(const float* grid, int n) {
    float max_temp = 0.0f;
    for (int i = 0; i < n * n; i++) {
        if (grid[i] > max_temp) {
            max_temp = grid[i];
        }
    }
    return max_temp;
}

int main() {
    size_t size = N * N * sizeof(float);
    float *h_grid = new float[N * N];

    for (int i = 0; i < N * N; i++) { h_grid[i] = 0.0f; }
    h_grid[(N / 2) * N + (N / 2)] = 1000.0f; // Initial heat source

    float *d_in, *d_out;
    cudaMalloc(&d_in, size);
    cudaMalloc(&d_out, size);
    cudaMemcpy(d_in, h_grid, size, cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE, 
                   (N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    std::cout << "Running " << STEPS << " simulation steps on CUDA...\n";

    auto start_time = std::chrono::high_resolution_clock::now();
    for (int step = 0; step < STEPS; ++step) {
        heatKernelNaive<<<numBlocks, threadsPerBlock>>>(d_in, d_out, N);
        float *temp = d_in;
        d_in = d_out;
        d_out = temp;
    }
    cudaDeviceSynchronize();
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> elapsed = end_time - start_time;

    // Bring results back for metrics check
    cudaMemcpy(h_grid, d_in, size, cudaMemcpyDeviceToHost);
    
    float peak_temp = findMaxTemperature(h_grid, N);

    std::cout << "----------------------------------------\n";
    std::cout << "CUDA Execution Time: " << elapsed.count() << " ms\n";
    std::cout << "Peak Grid Temperature after " << STEPS << " steps: " << peak_temp << " C\n";
    std::cout << "----------------------------------------\n";

    delete[] h_grid;
    cudaFree(d_in);
    cudaFree(d_out);
    return 0;
    
    
}

