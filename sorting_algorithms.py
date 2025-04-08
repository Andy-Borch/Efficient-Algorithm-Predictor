#Implement various sorting algorithms from the data in the csv
#Add run time for each algorithm on each peice of data to the csv file
import time
import random
import numpy as np
import pandas as pd

def bubble_sort(data):
    if isinstance(data, tuple) or isinstance(data, set):
        data = list(data)

    n = len(data)
    for i in range(n):
        for j in range(0, n - i - 1):
            if data[j] > data[j + 1]:
                data[j], data[j + 1] = data[j + 1], data[j]
    return data

def insertion_sort(data):
    if isinstance(data, tuple) or isinstance(data, set):
        data = list(data)
    return data

def merge_sort(data):
    if isinstance(data, tuple) or isinstance(data, set):
        data = list(data)
    return data

def quick_sort(data):
    if isinstance(data, tuple) or isinstance(data, set):
        data = list(data)
    return data

def heap_sort(data):
    if isinstance(data, tuple) or isinstance(data, set):
        data = list(data)
    return data

def radix_sort(data):
    if isinstance(data, tuple) or isinstance(data, set):
        data = list(data)
    return data

def bucket_sort(data):
    if isinstance(data, tuple) or isinstance(data, set):
        data = list(data)
    return data

def bubble_sort_data():
    #This data will eventually come from the csv file, this is just proof of concept for now
    array = random.sample(range(1, 10000), 1000)
    #TODO: Make each data structure type have different sizes/content to make more realistic to actual data we will use?
    tupleData = tuple(array)
    npArray = np.array(array)
    listData = list(array)
    pandasSeries = pd.Series(array)
    setData = set(array)

    timings = {}

    start = time.time()
    bubble_sort(array)
    timings['Array'] = time.time() - start

    start1 = time.time()
    bubble_sort(tupleData)
    timings['Tuple'] = time.time() - start1

    start2 = time.time()
    bubble_sort(npArray)
    timings['Numpy Array'] = time.time() - start2

    start3 = time.time()
    bubble_sort(listData)
    timings['List'] = time.time() - start3

    start4 = time.time()
    bubble_sort(pandasSeries)
    timings['Pandas Series'] = time.time() - start4

    start5 = time.time()
    bubble_sort(setData)
    timings['Set'] = time.time() - start5

    return timings

bubble_sort_results = bubble_sort_data()
df = pd.DataFrame(list(bubble_sort_results.items()), columns=['Data Structure Type', 'Time (seconds)'])
print("-----THIS IS ONLY FOR BUBBLE SORT AS PROOF OF CONCEPT. WILL EVENTUALLY SORT EACH DATA STRUCTURE TYPE WITH EVERY ALGORITHM-----")
print("Bubble Sort Results")
#TODO: Print fastest, then how much slower in % the rest are?
print(df)

#TODO: Make a for loop that changes data characteristics like length, std, etc and plot time vs certain characteristic? with a
#  scatterplot. y axis is time, x axis is a feature like length etc, each color dot represents a data structure type
    #make a scatter plot for each algorithm

