#include <iostream>
#include <vector>
#include <chrono>

const int N = 512; // Grid size NxN
const int STEPS = 1000;



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
