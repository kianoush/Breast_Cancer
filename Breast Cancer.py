"""
Breast Cancer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import os

"""
import data
"""
path = os.listdir()
print(path)
raw_data = pd.read_csv('breast_cancer_dataset.csv')

data = raw_data.iloc[:, :9].values
lables = raw_data.iloc[:, 9]
lables = np.where(lables == 2, 0, 1)
print(np.unique(lables))

"""
Split data to Train , Test and Valid
"""

X_train, X_test, Y_train, Y_test = train_test_split(data, lables, test_size=0.2, shuffle=True)
X_train, X_valid, Y_train1, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True)

X_train = torch.tensor(X_train).float()
X_test = torch.tensor(X_test).float()
X_valid = torch.tensor(X_valid).float()

Y_train = torch.tensor(Y_train1).long()
Y_test = torch.tensor(Y_test).long()
Y_valid = torch.tensor(Y_valid).long()

"""
Model
"""

num_class = 2
num_featurs = X_train.shape[1]
num_hiddenl = 10

model = torch.nn.Sequential(torch.nn.Linear(num_featurs, num_hiddenl),
                            torch.nn.ReLU(),
                            torch.nn.Linear(num_hiddenl, num_class)
)


"""
Loss
"""
loss = torch.nn.BCELoss()

"""
Optimizer
"""
optim = torch.optim.Adam(model.parameters(), lr=0.001)


num_sample_train = torch.tensor(X_train.shape[0])
num_sample_test = torch.tensor(X_test.shape[0])
num_sample_valid = torch.tensor(X_valid.shape[0])

num_epochs = 100
for epoch in range(num_epochs):
    optim = optim.zero_grad()
    Y_pred = model(X_train)
    loss_value = loss(Y_pred, Y_train)

    loss_value.backward()
    optim.step()

print('End!!')
