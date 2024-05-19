from module import *

USER_INDEX = 0
ITEM_INDEX = 1
RATING_INDEX = 2

class DataReader:
    def __init__(self, file_path = None):
        self.file_path = file_path

    def readCSV(self):
        if not self.file_path.endswith('.csv'):
            raise ValueError('Invalid file format')
        else:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                data = f.read()
                return data
            
    def readTXT(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            data = f.readlines()
            data = [line.strip('\n').split('\t') for line in data]
            data = [list(map(int, line)) for line in data]
            data = np.array(data)
            data = np.delete(data, (len(data[0]) -1), axis = 1)
            
            # Set the starter index to 0
            data[:,0: 2] = data[:, 0: 2] - 1
            return data
        
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