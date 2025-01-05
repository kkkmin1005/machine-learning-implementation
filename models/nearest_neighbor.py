import numpy as np

class NearestNeighbor:
    def __init__(self):
        pass

    def train(self, X, y):
        self.Xtr = X
        self.ytr = y
    
    def predict(self, X):
        # X is N x D where each row is an example we wish to predict label for
        num_test = X.shape[0] # shape[0]은 행의 수, shape[1]은 열의 수
        Ypred = np.zeros(num_test, dtype = self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i,:]), axis = 1) 
            # Xtr is M x D and X[i,:] is 1 x D, 브로드캐스팅을 통해 Xtr의 모든 행과 x의 차이를 구함 -> M x D의 결과가 나옴
            # sum, axis = 1 을 통해 각 행에 대해 모든 칼럼 값을 더함
            # M x 1의 최종 결과가 나옴옴
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred
