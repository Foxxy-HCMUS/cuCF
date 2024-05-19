from module import * 

def find_similarity_users(similarity_matrix, user, k):
    n_users = similarity_matrix.shape[0]
    similar_users = []
    
    # Kiểm tra xem user có hợp lệ không
    if user < 0 or user >= n_users:
        raise ValueError("Invalid user index")
    
    # Lặp qua tất cả các người dùng và chọn những người dùng có độ tương đồng với người dùng được chỉ định
    for other_user in range(n_users):
        if other_user != user:
            similarity = similarity_matrix[user][other_user]
            similar_users.append((other_user, similarity))
    
    # Sắp xếp danh sách các người dùng tương đồng theo độ tương đồng giảm dần
    similar_users.sort(key=lambda x: x[1], reverse=True)
    
    # Chọn k người dùng tương đồng hàng đầu
    if k > 0:
        similar_users = similar_users[:k]
    
    return similar_users

def predict(simility_matrix, normalized_matrix):
    n_users = normalized_matrix.shape[0]
    n_items = normalized_matrix.shape[1]
    predicted_matrix = np.zeros((n_users, n_items))
    
    for user in range(n_users):
        similar_user =  find_similarity_users(simility_matrix, user, 5)
        for item in range(n_items):
            if normalized_matrix[user][item] == 0:
                numerator = 0
                denominator = 0
                for u, similarity in similar_user:
                    if normalized_matrix[u][item] != 0:
                        numerator += similarity * normalized_matrix[u][item]
                        denominator += abs(similarity)
                if denominator != 0:
                    predicted_matrix[user][item] = numerator / denominator
    return predicted_matrix

# if rating predict > 0 -> return the list list(user, item_predict, rating_predict)
def recommend(predict_matrix, avg_users):
    n_users = predict_matrix.shape[0]
    n_items = predict_matrix.shape[1]
    recommend_list = []
    for user in range(n_users):
        for item in range(n_items):
            if predict_matrix[user][item] > 0:
                recommend_list.append((user, item, predict_matrix[user][item] + avg_users[user]))
    return recommend_list
    