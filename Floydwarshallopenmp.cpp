
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <omp.h>

#define INF 1000000000
#define N 500


// Function to apply the Floyd-Warshall algorithm on the matrix
static void floyd_warshall(const int num_nodes, int *graph) {
    #pragma omp parallel
    {
        for (int k=0; k<num_nodes; k++) {
            #pragma omp for collapse(4)
            for(int src=0; src<num_nodes; src++) {
                for(int dest=0; dest<num_nodes; dest++) {
                    int other = graph[src*num_nodes+k] + graph[k*num_nodes+dest];
                    if (graph[src*num_nodes+dest] > other) {
                        graph[src*num_nodes+dest] = other;
                    }
                }
            }
        }
    }
}

int main() {
    // Seed random number generator
    srand(time(NULL));

    // Allocate memory for the matrix
    int* matrix = (int*) malloc(N*N*sizeof(int));

    // Generate random values for the matrix
    for (int i=0; i<N; i++) {
        for (int j=0; j<N; j++) {
          if(i==j){
            matrix[i*N+j] = 0;
          }
          else {
                matrix[i*N+j] = (rand() % 10) + 1;
                if (matrix[i*N+j] > INF) {
                    matrix[i*N+j] = INF;
                }
            }
        }
        }
    }

    // Apply the Floyd-Warshall algorithm and time it
    double start_time = omp_get_wtime();
    floyd_warshall(N, matrix);
    double end_time = omp_get_wtime();

    // Print the time taken for the algorithm to complete
    printf("Time taken: %f seconds\n", end_time - start_time);

    // Free memory
    free(matrix);

    return 0;
}
