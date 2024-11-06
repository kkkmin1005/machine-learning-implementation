# logistic Regressison model

# what is logistic regression
# Given features and label, predict label when features are provided as input

# 0. import library
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# 1. load datasets
df = pd.read_csv(r'C:\Users\kangmin\Desktop\ai-project\datasets\iris.csv')

input_data = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
target_data = df[['species']]

train_input, test_input, train_target, test_target = train_test_split(input_data, target_data, random_state=42, test_size=0.2)

# 2. implement model
class logisticRegression():
    def __init__(self):
        self.x = None
        self.y = None
        self.w = None
        self.alpha = None
    
    def addX0(self):
        self.x = np.column_stack(self.x, np.array([1 for _ in range(len(self.x))]))
    
    def forward(self):
        self.p = self.x @ self.w

    def update(self):
        gradient_of_loss = 
