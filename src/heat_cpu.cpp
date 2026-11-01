

void simulateCPU(std::vector<float>& grid, std::vector<float>& new_grid) {
    for (int step = 0; step < STEPS; ++step) {
        for (int i = 1; i < N - 1; ++i) {
            for (int j = 1; j < N - 1; ++j) {
                new_grid[i * N + j] = 0.25f * (
                    grid[(i - 1) * N + j] +
                    grid[(i + 1) * N + j] +
                    grid[i * N + (j - 1)] +
                    grid[i * N + (j + 1)]
                );
            }
        }
        grid = new_grid;
    }
}

int main() {
    std::vector<float> grid(N * N, 0.0f);
    std::vector<float> new_grid(N * N, 0.0f);

    grid[(N / 2) * N + (N / 2)] = 1000.0f;

    auto start = std::chrono::high_resolution_clock::now();
    simulateCPU(grid, new_grid);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::milli> elapsed = end - start;
    std::cout << "CPU Execution Time for " << STEPS << " steps: " << elapsed.count() << " ms\n";
    return 0;
}
