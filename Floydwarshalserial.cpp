
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define INF 1000000000

void floyd_warshall(int **dist, int n) {
    int i, j, k;
    for (k = 0; k < n; k++) {
        for (i = 0; i < n; i++) {
            for (j = 0; j < n; j++) {
                if (dist[i][k] + dist[k][j] < dist[i][j]) {
                    dist[i][j] = dist[i][k] + dist[k][j];
                }
            }
        }
    }
}

int main(int argc, char *argv[]) {
    int n = atoi(argv[1]);
    int **dist = (int **) malloc(n * sizeof(int *));
    for (int i = 0; i < n; i++) {
        dist[i] = (int *) malloc(n * sizeof(int));
    }

    // Fill the matrix with random numbers
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (i == j) {
                dist[i][j] = 0;
            } else {
                dist[i][j] = (rand() % 10) + 1;
                if (dist[i][j] > INF) {
                    dist[i][j] = INF;
                }
            }
        }
    }

 
