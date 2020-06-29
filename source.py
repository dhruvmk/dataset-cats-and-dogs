# Importing appropriate dependancies
from tensorflow import keras
from keras.datasets import cifar10
import numpy as np
from sklearn.utils import shuffle

# Loading the data
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

# Merging our train and test data to make filtering easier
images = np.concatenate((train_images, test_images))
labels = np.concatenate((train_labels, test_labels))

# Filtering
# Here is a dictionary of the indices of the labels to the class names:
dictionary = {0:'airplane',
              1:'automobile',
              2:'bird',
              3:'cat',
              4:'deer',
              5:'dog',
              6:'frog',
              7:'horse',
              8:'ship',
              9:'truck'}

# Since we want the images of cats, we will look for when the labels = 3:
cat_indices = np.where(labels==3)[0]
cat_labels = labels[cat_indices]
cat_images = images[cat_indices]

# We also want the dogs, so we will look for when the labels = 5:
dog_indices = np.where(labels==5)[0]
dog_labels = labels[dog_indices]
dog_images = images[dog_indices]

# Prepare binary labels
y_cats = [0 for c in cat_labels]
y_dogs = [1 for d in dog_labels]

# Merge and shuffle
x = np.concatenate((cat_images, dog_images))
y = np.concatenate((y_cats, y_dogs))
x, y = shuffle(x, y, random_state=42)

# Finally, saving our data into binary format
np.save('x.npy', x)
np.save('y.npy', y)

