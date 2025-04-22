#Generate lists, tuples, linkedLists
#Put into csv file
#include size, number of unique elements, etc
#maybe create a function that determines initial sortedness without sorting

import numpy as np
from numpy.random import default_rng

min_size, max_size = 100, 1000
num_points = 1000

rng = np.random.default_rng()
size_features = np.random.randint(min_size, max_size, size=(1, 1000)) 
print(size_features)

for x in range(num_points):
    curr_size = 
    floats = default_rng(42).random()