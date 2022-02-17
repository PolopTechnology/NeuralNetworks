import numpy as np
import random as r
from time import sleep
from tqdm import tqdm

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

x_train = [[0, 0], [0, 1], [1, 0], [1, 1]]
y_train = [0, 1, 1, 0]
W = np.random.uniform(0.1, 0.9, size=(3, 2))
B = np.random.uniform(0.1, 0.9, size=(3, 1))

lr = 0.003

def feedforward(X, O):
    Z1 = ((X[0] * W[0, 0]) + (X[1] * W[0, 1])) + B[0]
    Z2 = ((X[0] * W[1, 0]) + (X[1] * W[1, 1])) + B[1]
    AZ1 = sigmoid(Z1)
    AZ2 = sigmoid(Z2)
    Y = (((AZ1 * W[2,0]) + (AZ2 * W[2,1])) + B[2])
    AY = sigmoid(Y)
    E = (O - AY) ** 2
    DE = 2 * (AY - O)
    E2 = (W[0] * DE) + (W[1] * DE)
    #print(E)
    return AZ1, AZ2, Y, AY, E, E2, DE

def feedbackward(AZ1, AZ2, Y, AY, E, E2, DE):
    DW3 = DE * (AY * (1 - AY)) * np.array([AZ1, AZ2])
    DW2 = E2[1] * (AZ2 * (1 - AZ2)) * X
    DW1 = E2[0] * (AZ1 * (1 - AZ1)) * X
    DB3 = DE * (AY * (1 - AY))
    DB2 = E2[1] * (AZ2 * (1 - AZ2))
    DB1 = E2[0] * (AZ1 * (1 - AZ1))
    W[2] -= lr * DW3[0]
    W[1] -= lr * DW2[0]
    W[0] -= lr * DW1[0]
    B[2] -= lr * DB3
    B[1] -= lr * DB2
    B[0] -= lr * DB1

for i in tqdm(range(2_000_000)): 
    X = (r.choice(x_train))
    O = [y_train[i] for i, arr in enumerate(x_train) if arr == X]
    AZ1, AZ2, Y, AY, E, E2, DE = feedforward(X, O)
    feedbackward(AZ1, AZ2, Y, AY, E, E2, DE)

#for i in range(4):
    #X = np.array(x_train[i])
    #O = y_train[i]
    #for i in tqdm.tqdm(range(10_000)):
        #AZ1, AZ2, Y, AY, E, E2 = feedforward()
        #feedbackward(AZ1, AZ2, Y, AY, E, E2)
    #print(E)
    #sleep(1)

for i in range(4):
    I = np.array(x_train[i])
    U = y_train[i]
    AZ1, AZ2, Y, AY, E, E2, DE = feedforward(I, U)
    print(I, U, AY)