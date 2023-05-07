from mpi4py import MPI
import numpy as np

# Initialize MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Generate random adjacency matrix
n = 1000  # matrix size
if rank == 0:
    A = np.random.randint(1, 100, size=(n, n))
else:
    A = None

# Scatter rows of adjacency matrix
local_n = n // size
local_A = np.zeros((local_n, n), dtype=int)
comm.Scatter(A, local_A, root=0)

# Initialize matrix of shortest paths
D = np.copy(local_A)

# Compute shortest paths
for k in range(n):
    # Broadcast the k-th row of D to all processes
    k_row = np.zeros((1, n), dtype=int)
    if rank == (k // local_n):
        k_row = np.copy(D[k % local_n])
    comm.Bcast(k_row, root=(k // local_n))

    # Compute shortest paths for subset of rows
    for i in range(local_n):
        for j in range(n):
            D[i][j] = min(D[i][j], D[i][k] + k_row[j])

    # Combine rows of D
    comm.Allgather(local_A, D)

# Gather rows of D
if rank == 0:
    D = np.zeros((n, n), dtype=int)
comm.Gather(local_A, D, root=0)

if rank == 0:
    print(D)


