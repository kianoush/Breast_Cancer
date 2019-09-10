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
Split to Train , Test and Valid
"""



print('End!!')
