import numpy as np

# loading data
x = np.load(x.npy)
y = np.load(y.npy)

# normalizing
x = x/255

# splitting data
from sklearn.model_selection import train_test_split
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2)
