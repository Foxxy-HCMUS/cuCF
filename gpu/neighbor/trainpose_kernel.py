# import numba as nb
from module import *
TILE_DIM = 32

@cuda.jit
def kernel_tranpose(odata, idata, width: int, height: int):
    '''
    This function transposes the input matrix 'idata' and stores the result in 'odata'.
    '''
    s_array = cuda.shared.array(shape=(TILE_DIM, TILE_DIM))
    x = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.y

    if (x < width and y < height):
        s_array[cuda.threadIdx.y, cuda.threadIdx.x] = idata[y * width + x]
    cuda.syncthreads()


    # tranpose.
    x = cuda.blockIdx.x * TILE_DIM + cuda.threadIdx.x
    y = cuda.blockIdx.y * TILE_DIM + cuda.threadIdx.y

    if (y < width and x < height):
        odata[x * height + y] = s_array[cuda.threadIdx.x, cuda.threadIdx.y]