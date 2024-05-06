from module import *

USER_INDEX = 0
ITEM_INDEX = 1
RATING_INDEX = 2

def get_utility_matrix(data):
    n_users = data[:, USER_INDEX].max() + 1
    n_items = data[:, ITEM_INDEX].max() + 1

    utility_matrix = np.zeros((n_users, n_items))
    for row in data:
        user, item, rating = row
        utility_matrix[user][item] = rating
    return utility_matrix

def serial_mean(utility_matrix):
    n_items = utility_matrix.shape[1]
    n_users = utility_matrix.shape[0]
    avg_users = np.zeros(n_users)
    for u in range(n_users):
        sum = 0
        countNonZero = 0
        for i in range(n_items):
            if utility_matrix[u][i] > 0:
                sum += utility_matrix[u][i]
                countNonZero += 1
        if countNonZero > 0:
            avg = sum / countNonZero
            avg_users[u] = avg
            for i in range(n_items):
                # Bỏ qua các giá trị 0, chỉ trừ trung bình đối với các giá trị đã được đánh giá.
                if utility_matrix[u][i] > 0:
                    utility_matrix[u][i] -= avg
        else:
            avg = 0

    return utility_matrix, avg_users
        

def build_similarity_matrix_cosine_similarity(mean_matrix):
    n_items = mean_matrix.shape[1]  
    n_users = mean_matrix.shape[0]
    similarity_matrix = np.zeros((n_users, n_users))
    similarity = 0
    for us in range(n_users):
        for other_u in range(n_users):
            if us == other_u:
                similarity = 1
            elif us < other_u:
                numerator = 0
                sum_us = 0
                sum_other_u = 0
                for i in range(n_items):
                    numerator+= mean_matrix[us][i] * mean_matrix[other_u][i]
                    sum_us += mean_matrix[us][i]**2
                    sum_other_u += mean_matrix[other_u][i]**2
                similarity = numerator/ (np.sqrt(sum_us) * np.sqrt(sum_other_u)) 
            similarity_matrix[us][other_u] = similarity
            similarity_matrix[other_u][us] = similarity
    
    
    return similarity_matrix