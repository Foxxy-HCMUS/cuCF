import dataReader
import predict
from similarityMatrix import *
import numpy as np
import json

if __name__ == '__main__':
    data = dataReader.DataReader('data/ml-100k/ua.base').readTXT()
    utility_matrix = get_utility_matrix(data)
    np.savetxt('out/utility.csv', utility_matrix, delimiter=',')

    print(utility_matrix[0][1448])
    
    
    start = get_time()
    mean_matrix, avg_users = serial_mean(utility_matrix)
    end = get_time()
    print("Time to calculate mean matrix: ", end - start)

    np.savetxt('out/mean_matrix.csv', mean_matrix, delimiter=',')
    np.savetxt('out/avg_user.csv', avg_users, delimiter=',')

    start = get_time()
    similarity_matrix = build_similarity_matrix_cosine_similarity(mean_matrix)
    end = get_time()
    print("Time to calculate similarity matrix: ", end - start)

    np.savetxt('out/similarity_matrix.csv', similarity_matrix, delimiter=',')
    print(similarity_matrix.shape)

    # predict
    start = get_time()
    predicted_matrix = predict.predict(similarity_matrix, mean_matrix)
    end = get_time()
    print("Time to predict: ", end - start)
    np.savetxt('out/predicted_matrix.csv', predicted_matrix, delimiter=',')

    # recommend
    recommend_list = predict.recommend(predicted_matrix, avg_users)
    with open('out/recommend.json', 'w') as f:
        json.dump(recommend_list, f, indent=4)