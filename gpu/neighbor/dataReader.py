import numpy as np

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