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
    pass

def merge_sort(data):
    pass

def quick_sort(data):
    pass

def heap_sort(data):
    pass

def radix_sort(data):
    pass

def bucket_sort(data):
    pass

def sort_data():
    #This data will eventually come from the csv file, this is just proof of concept for now
    array = random.sample(range(1, 10000), 1000)
    tupleData = tuple(array)
    npArray = np.array(array)
    listData = list(array)
    pandasSeries = pd.Series(array)
    setData = set(array)

    timings = {}

    start = time.time()
    bubble_sort(array)
    timings['Bubble Sort'] = time.time() - start

    start1 = time.time()
    bubble_sort(tupleData)
    timings['Tuple bubble'] = time.time() - start1

    start2 = time.time()
    bubble_sort(npArray)
    timings['NP array'] = time.time() - start2

    start3 = time.time()
    bubble_sort(listData)
    timings['list'] = time.time() - start3

    start4 = time.time()
    bubble_sort(pandasSeries)
    timings['Pandas'] = time.time() - start4

    start5 = time.time()
    bubble_sort(setData)
    timings['set'] = time.time() - start5

    return timings

results = sort_data()
df = pd.DataFrame(list(results.items()), columns=['Algorithm', 'Time (seconds)'])
print("-----THIS IS ONLY FOR BUBBLE SORT AS PROOF OF CONCEPT. WILL EVENTUALLY SORT EACH DATA STRUCTURE TYPE WITH EVERY ALGORITHM-----")
print(df)
