"""
Breast Cancer
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import os

"""
import data
"""
path = os.listdir()
print(path)
raw_data = pd.read_csv('breast_cancer_dataset.csv')

data = raw_data.iloc[:, :9].values
lables = raw_data.iloc[:, 9]

"""
Split data to Train , Test and Valid
"""

X_train, X_test,Y_train, Y_test = train_test_split(data, lables, test_size=0.3, shuffle=True)
X_train, X_valid,Y_train, Y_valid = train_test_split(X_train, Y_train, test_size=0.2, shuffle=True)



"""

"""
print('End!!')
