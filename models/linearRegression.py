# linear regression model

# what is linear regression
# Given data with two variables, x and y, predict y when x is provided as input

# 0. import library
import numpy as np
import pandas as pd

# 1. load dataset
df = pd.read_csv('datasets/Delhi house data.csv')

x = df['Area']
y = df['Price']

# 2. split dataset into train and test
train_x = x[:1000].to_numpy()
train_y = y[:1000].to_numpy()
test_x = x[1000:].to_numpy()
test_y = y[1000:].to_numpy()

# 3. implement model
class linearRegression():

    # initialize model's x,y,w
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.alpha = None

    # add x0 for bias weight
    def addX0(self):
        self.x = np.column_stack((self.x, np.array([1 for _ in range(len(self.x))])))

    # train model
    def train(self, x, y, epoch, alpha):
        self.x = x
        self.y = y
        self.alpha = alpha
        self.addX0()
        self.w = np.array([0 for _ in range(len(self.x[0]))])

        for i in range(epoch):
            self.forward()
            self.update()
            print(f"-----epoch{i+1}-----")
            print(self.score(self.x, self.y))

    # predict y with x and w
    def forward(self):
        self.p = self.x @ self.w
    
    # update weight
    def update(self):
        gradient_of_loss = self.x.T @ (self.p - self.y) / len(self.x)
        self.w = self.w - self.alpha * (gradient_of_loss)

    #get score of model
    def score(self, x, y):
        top = (self.p - self.y).T @ (self.p - self.y)
        bottom = (np.mean(self.y) - self.y).T @ (np.mean(self.y) - self.y)
        return (1- (top/bottom)) * 100
    
model = linearRegression()
model.train(epoch=10, alpha=0.1 ** 7, x= train_x, y = train_y)



        

    

    

    


