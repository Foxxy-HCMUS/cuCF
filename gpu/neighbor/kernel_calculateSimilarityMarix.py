from module import *

@cuda.jit
def caculate_similarity_matrix_kernel1(mean_matrix, similarity_matrix):
    us = cuda.grid(1)
    if us < mean_matrix.shape[0]:
        for other_u in range(mean_matrix.shape[0]):
            if us == other_u:
                similarity = 1
            elif us < other_u:
                numerator = 0
                sum_us = 0
                sum_other_u = 0
                for i in range(mean_matrix.shape[1]):
                    numerator+= mean_matrix[us][i] * mean_matrix[other_u][i]
                    sum_us += mean_matrix[us][i]**2
                    sum_other_u += mean_matrix[other_u][i]**2
                similarity = numerator/ (np.sqrt(sum_us) * np.sqrt(sum_other_u)) 
            similarity_matrix[us][other_u] = similarity
            similarity_matrix[other_u][us] = similarity