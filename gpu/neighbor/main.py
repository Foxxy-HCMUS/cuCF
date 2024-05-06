import dataReader
from similarityMatrix import *
import numpy as np
import json

if __name__ == '__main__':
    data = dataReader.DataReader('data/ml-100k/u.data').readTXT()
    utility_matrix = get_utility_matrix(data)
    np.savetxt('out/utility.csv', utility_matrix, delimiter=',')
    mean_matrix, avg_users = serial_mean(utility_matrix)
    np.savetxt('out/mean_matrix.csv', mean_matrix, delimiter=',')
    np.savetxt('out/avg_user.csv', avg_users, delimiter=',')
    similarity_matrix = build_similarity_matrix_cosine_similarity(mean_matrix)
    np.savetxt('out/similarity_matrix.csv', similarity_matrix, delimiter=',')
    print(similarity_matrix.shape)