#Tests the usage of KFolds

import numpy as np
from sklearn.model_selection import KFold

X = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
y = np.array([1, 2, 3, 4])

print(X)

kf = KFold(n_splits = 2)

#print(kf.get_n_splits(X))

for train_index, test_index in kf.split(X):
    print('Train:  %s, Test: %s' % (train_index, test_index))