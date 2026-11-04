#include <iostream>
#include <vector>
#include <chrono>

// Note : chrono is a c++ library used to measure work with time

const int N = 512; // Grid size NxN
const int STEPS = 1000;

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

