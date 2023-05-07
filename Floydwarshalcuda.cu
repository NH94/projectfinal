#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>

#define N 1000
#define INF 1000000000

__global__ void floyd_warshall(int *dist, int k) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (i < N && j < N) {
        int ij = dist[i * N + j];
        int ik = dist[i * N + k];
        int kj = dist[k * N + j];
        if (ik + kj < ij) {
            dist[i * N + j] = ik + kj;
        }
    }
}

void init(int *dist, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                dist[i * n + j] = 0;
            } else {
                dist[i * n + j] = (rand() % 10) + 1;
                if (dist[i * n + j] > INF) {
                    dist[i * n + j] = INF;
                }
            }
        }
    }
}

void floyd_warshall_cuda(int *dist, int n) {
    int *dist_d;
    size_t size = n * n * sizeof(int);
    cudaMalloc(&dist_d, size);
    cudaMemcpy(dist_d, dist, size, cudaMemcpyHostToDevice);

    dim3 threads_per_block(4, 4);
    dim3 num_blocks((N + threads_per_block.x - 1) / threads_per_block.x,
                    (N + threads_per_block.y - 1) / threads_per_block.y);

    for (int k = 0; k < n; k++) {
        floyd_warshall<<<num_blocks, threads_per_block>>>(dist_d, k);
    }

    cudaMemcpy(dist, dist_d, size, cudaMemcpyDeviceToHost);
    cudaFree(dist_d);
}

int main() {
    int dist[N][N];
    init(&dist[0][0], N);

    struct timeval start_time, end_time;
    gettimeofday(&start_time, NULL);

    floyd_warshall_cuda(&dist[0][0], N);

    gettimeofday(&end_time, NULL);
    long long elapsed_time = (end_time.tv_sec - start_time.tv_sec) * 1000000LL + (end_time.tv_usec - start_time.tv_usec);
    printf("Elapsed time: %lld microseconds\n", elapsed_time);

    return 0;
}
