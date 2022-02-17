import numpy as np
import random as r
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 0, 0, 1]
W = np.random.uniform(0.1, 0.9, size=(1, 2))
B = np.random.uniform(0.1, 0.9, size=(1, 1))
lr = 0.03

def feedforward(X, O):
    Z = ((X[0] * W[0,0]) + (X[1] * W[0,1])) + B[0]
    A = sigmoid(Z)
    E = (O - A) ** 2
    #print(E)
    DE = 2 * (A - O)
    return X, DE, A

def feedbackward(X, DE, A):
    DW = DE * (A * (1 - A)) * X
    DB = DE * (A * (1 - A))
    W[0] -= lr * DW
    B[0] -= lr * DB

for i in tqdm(range(100_000)): 
    X = (r.choice(x_train))
    O = [y_train[i] for i, arr in enumerate(x_train) if arr == X]
    X, DE, A = feedforward(X, O)
    feedbackward(X, DE, A)

for i in range(4):
    I = np.array(x_train[i])
    U = y_train[i]
    X, DE, A = feedforward(I, U)
    print(I, U, A)