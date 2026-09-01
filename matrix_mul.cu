// matrix_mul.cu
// DATA 266 - Homework 1 - CUDA C tiled matrix multiplication
// Student: Drashti Shah | Full SID: 013852046 | SID4 = 2046, SEED = 2046
//
// Build:
//   nvcc -O3 -std=c++17 -Xcompiler -fopenmp -lineinfo matrix_mul.cu -o matrix_mul
//
// Full benchmark (writes cuda_metrics.csv, prints the timing table):
//   ./matrix_mul 256 1024 4096
//
// Single-size GPU-only run for a profiler to attach to (no CPU reference computed):
//   ncu --set basic ./matrix_mul --profile 1024
//
// -----------------------------------------------------------------------------
// Blocks and threads: TILE = 16, so every block is a 2D block of 16 x 16 = 256
// threads. The grid is 2D with ceil(n/16) blocks per dimension, so each block
// owns one 16x16 tile of output matrix C and each thread owns exactly one
// output element. To compute it, the block walks matching 16x16 tiles of A
// (moving right) and B (moving down): every thread loads one A-value and one
// B-value into shared memory (As/Bs), the block calls __syncthreads() so every
// thread's loads are visible, then every thread reuses those 16 shared values
// for 16 multiply-accumulate steps, and __syncthreads() again before the next
// tile overwrites shared memory. Loop condition is `t < numTiles`
// (numTiles = tileCount). Boundary checks (row/col/aCol/bRow < n) keep results
// correct even for sizes that are not exact multiples of 16.
// -----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <chrono>
#include <vector>
#include <string>
#include <fstream>
#include <omp.h>
#include <cuda_runtime.h>

#define SEED_VALUE 2046
#define TILE 16
#define CPU_REPEATS 3
#define GPU_REPEATS 10

#define CUDA_CHECK(call)                                                      \
    do {                                                                      \
        cudaError_t err = (call);                                             \
        if (err != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,  \
                    __LINE__, cudaGetErrorString(err));                       \
            exit(EXIT_FAILURE);                                               \
        }                                                                     \
    } while (0)

__global__ void tiled_matrix_multiply(const float* A, const float* B, float* C, int n) {
    __shared__ float As[TILE][TILE];
    __shared__ float Bs[TILE][TILE];

    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    float acc = 0.0f;
    int tileCount = (n + TILE - 1) / TILE;

    for (int t = 0; t < tileCount; ++t) {
        int aCol = t * TILE + threadIdx.x;
        int bRow = t * TILE + threadIdx.y;

        As[threadIdx.y][threadIdx.x] = (row < n && aCol < n) ? A[(size_t)row * n + aCol] : 0.0f;
        Bs[threadIdx.y][threadIdx.x] = (bRow < n && col < n) ? B[(size_t)bRow * n + col] : 0.0f;

        __syncthreads(); // all loads visible before compute

#pragma unroll
        for (int k = 0; k < TILE; ++k) {
            acc += As[threadIdx.y][k] * Bs[k][threadIdx.x];
        }
        __syncthreads(); // all compute done before next tile overwrites shared mem
    }

    if (row < n && col < n) {
        C[(size_t)row * n + col] = acc;
    }
}

static void cpu_matrix_multiply(const std::vector<float>& A, const std::vector<float>& B,
                                 std::vector<float>& C, int n) {
    // i-k-j loop order (instead of naive i-j-k): both B and C are accessed
    // row-wise (stride-1) in the innermost loop, which is cache-friendly and
    // auto-vectorizes well with -O3. The naive i-j-k order walks B column-wise
    // in the inner loop, which is close to the worst possible access pattern
    // and made N=4096 take many minutes instead of seconds.
    std::fill(C.begin(), C.end(), 0.0f);
#pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        for (int k = 0; k < n; ++k) {
            float a_ik = A[(size_t)i * n + k];
            const float* b_row = &B[(size_t)k * n];
            float* c_row = &C[(size_t)i * n];
            for (int j = 0; j < n; ++j) {
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

static void fill_matrix(std::vector<float>& M, unsigned int seed) {
    srand(seed);
    for (auto& v : M) v = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
}

struct RunResult {
    int n;
    double cpu_ms = 0.0;
    double kernel_ms = 0.0;
    double transfer_ms = 0.0;
    double end_to_end_ms = 0.0;
    double speedup = 0.0;
    double max_error = -1.0; // -1 means "not computed" (GPU-only profiling mode)
};

// with_cpu_reference=false skips the CPU triple loop entirely, for GPU-only
// profiler attachment where the slow CPU pass would otherwise get profiled too.
static RunResult benchmark(int n, bool with_cpu_reference) {
    size_t elems = (size_t)n * n;
    size_t bytes = elems * sizeof(float);

    std::vector<float> hA(elems), hB(elems), hC(elems), hRef;
    fill_matrix(hA, SEED_VALUE + n);
    fill_matrix(hB, SEED_VALUE + n + 1);

    RunResult result;
    result.n = n;

    if (with_cpu_reference) {
        printf("  Running CPU reference for N=%d (%d repeats)...\n", n, CPU_REPEATS);
        fflush(stdout);
        hRef.resize(elems);
        double cpu_total = 0.0;
        for (int r = 0; r < CPU_REPEATS; ++r) {
            auto cpu_start = std::chrono::high_resolution_clock::now();
            cpu_matrix_multiply(hA, hB, hRef, n);
            auto cpu_end = std::chrono::high_resolution_clock::now();
            cpu_total += std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();
        }
        result.cpu_ms = cpu_total / CPU_REPEATS;
    }

    float *dA = nullptr, *dB = nullptr, *dC = nullptr;
    CUDA_CHECK(cudaMalloc(&dA, bytes));
    CUDA_CHECK(cudaMalloc(&dB, bytes));
    CUDA_CHECK(cudaMalloc(&dC, bytes));

    dim3 threads(TILE, TILE);
    dim3 blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);

    cudaEvent_t evStart, evStop, evH2DStart, evH2DStop, evD2HStart, evD2HStop;
    CUDA_CHECK(cudaEventCreate(&evStart));
    CUDA_CHECK(cudaEventCreate(&evStop));
    CUDA_CHECK(cudaEventCreate(&evH2DStart));
    CUDA_CHECK(cudaEventCreate(&evH2DStop));
    CUDA_CHECK(cudaEventCreate(&evD2HStart));
    CUDA_CHECK(cudaEventCreate(&evD2HStop));

    // One untimed warm-up launch.
    CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
    tiled_matrix_multiply<<<blocks, threads>>>(dA, dB, dC, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    int repeats = with_cpu_reference ? GPU_REPEATS : 1; // profiler mode: single GPU-only pass
    double kernel_total = 0.0, transfer_total = 0.0;
    for (int r = 0; r < repeats; ++r) {
        float h2d_ms = 0.0f, kernel_ms = 0.0f, d2h_ms = 0.0f;

        CUDA_CHECK(cudaEventRecord(evH2DStart));
        CUDA_CHECK(cudaMemcpy(dA, hA.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dB, hB.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(evH2DStop));
        CUDA_CHECK(cudaEventSynchronize(evH2DStop));
        CUDA_CHECK(cudaEventElapsedTime(&h2d_ms, evH2DStart, evH2DStop));

        CUDA_CHECK(cudaEventRecord(evStart));
        tiled_matrix_multiply<<<blocks, threads>>>(dA, dB, dC, n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(evStop));
        CUDA_CHECK(cudaEventSynchronize(evStop));
        CUDA_CHECK(cudaEventElapsedTime(&kernel_ms, evStart, evStop));

        CUDA_CHECK(cudaEventRecord(evD2HStart));
        CUDA_CHECK(cudaMemcpy(hC.data(), dC, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaEventRecord(evD2HStop));
        CUDA_CHECK(cudaEventSynchronize(evD2HStop));
        CUDA_CHECK(cudaEventElapsedTime(&d2h_ms, evD2HStart, evD2HStop));

        kernel_total += kernel_ms;
        transfer_total += (h2d_ms + d2h_ms);
    }

    result.kernel_ms = kernel_total / repeats;
    result.transfer_ms = transfer_total / repeats;
    result.end_to_end_ms = result.kernel_ms + result.transfer_ms;

    if (with_cpu_reference) {
        result.speedup = result.cpu_ms / result.end_to_end_ms;
        double max_error = 0.0;
        for (size_t i = 0; i < elems; ++i) {
            double diff = std::fabs((double)hC[i] - (double)hRef[i]);
            if (diff > max_error) max_error = diff;
        }
        result.max_error = max_error;
    }

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(evStart));
    CUDA_CHECK(cudaEventDestroy(evStop));
    CUDA_CHECK(cudaEventDestroy(evH2DStart));
    CUDA_CHECK(cudaEventDestroy(evH2DStop));
    CUDA_CHECK(cudaEventDestroy(evD2HStart));
    CUDA_CHECK(cudaEventDestroy(evD2HStop));

    return result;
}

int main(int argc, char** argv) {
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("GPU: %s\n", prop.name);
    printf("Block dimensions: %d x %d = %d threads per block\n", TILE, TILE, TILE * TILE);

    if (argc >= 3 && std::string(argv[1]) == "--profile") {
        int n = std::atoi(argv[2]);
        dim3 threads(TILE, TILE);
        dim3 blocks((n + TILE - 1) / TILE, (n + TILE - 1) / TILE);
        printf("Profiling target: N=%d, grid=(%d,%d,1), block=(%d,%d,1), total threads=%zu\n",
               n, blocks.x, blocks.y, threads.x, threads.y, (size_t)blocks.x * blocks.y * threads.x * threads.y);
        RunResult r = benchmark(n, /*with_cpu_reference=*/false);
        printf("Profiling run N=%d complete (GPU-only, no CPU reference computed). "
               "kernel=%g ms, transfer=%g ms\n", r.n, r.kernel_ms, r.transfer_ms);
        return 0;
    }

    std::vector<int> sizes;
    for (int i = 1; i < argc; ++i) sizes.push_back(std::atoi(argv[i]));
    if (sizes.empty()) sizes = {256, 1024, 4096};

    std::ofstream csv("cuda_metrics.csv");
    csv << "Matrix size,CPU (ms),GPU kernel (ms),H2D+D2H (ms),GPU end-to-end (ms),Speedup,Max absolute error\n";

    printf("\n%-6s %-14s %-14s %-14s %-16s %-10s %-14s\n",
           "N", "CPU(ms)", "Kernel(ms)", "Transfer(ms)", "EndToEnd(ms)", "Speedup", "MaxAbsErr");

    for (int n : sizes) {
        RunResult r = benchmark(n, /*with_cpu_reference=*/true);
        printf("%-6d %-14.6g %-14.6g %-14.6g %-16.6g %-10.6g %-14.6g\n",
               r.n, r.cpu_ms, r.kernel_ms, r.transfer_ms, r.end_to_end_ms, r.speedup, r.max_error);
        csv << r.n << "," << r.cpu_ms << "," << r.kernel_ms << "," << r.transfer_ms << ","
            << r.end_to_end_ms << "," << r.speedup << "," << r.max_error << "\n";
    }
    csv.close();
    printf("\nWrote cuda_metrics.csv\n");
    return 0;
}
