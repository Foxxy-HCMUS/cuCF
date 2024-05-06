from module import *

# serial_mean + compute the norm of the matrix
@cuda.jit
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

