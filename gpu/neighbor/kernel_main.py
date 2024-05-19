from module import *
import dataReader
from similarityMatrix import *

@cuda.jit(device=True)
def kernel_get_average_and_norm(R, cols: int, rows: int, avg, norm):
    # row: n_users, col: n_items
    bix = cuda.blockIdx.x
    bdx = cuda.blockDim.x
    tix = cuda.threadIdx.x

    # thread ID -> item index
    # 1 block -> 1 item
    tid = bix * bdx + tix
    if tid < cols:
        countNonZero = 0
        sum = 0
        avgThread = 0
        
        # With per user: 
        for i in range(rows):
            # Who is the user that rated this item
            
            if (R[i*cols + tid] > 0):
                sum += R[i*cols + tid]
                countNonZero += 1
            
        if countNonZero > 0:
            avgThread = sum / countNonZero
        else:
            avgThread = 0
        
        if (tid < cols):
            avg[tid] = avgThread

        sum = 0
        for i in range(rows):
            if (R[i*cols + tid] > 0):
                sum += (R[i*cols + tid] - avgThread)**2
        
        if (tid < cols):
            norm[tid] = sum

if __name__ == "__main__":
    data = dataReader.DataReader('data/ml-100k/ua.base').readTXT()
    utility_matrix = get_utility_matrix(data)
    np.savetxt('out/utility.csv', utility_matrix, delimiter=',')

    # Step 1: Calculate the mean matrix by kernel 
    start_time = get_time()
    R = utility_matrix
    n_users = R.shape[0]
    n_items = R.shape[1]
    avg = np.zeros(n_items)
    norm = np.zeros(n_items)
    block_size = 256
    grid_size = (n_items + block_size - 1) // block_size
    kernel_get_average_and_norm[grid_size, block_size](R, n_items, n_users, avg, norm)
    end_time = get_time()
    print("Time to calculate the mean matrix: ", end_time - start_time)
    np.savetxt('out/avg.csv', avg, delimiter=',')
    np.savetxt('out/norm.csv', norm, delimiter=',')
    print("Done!")