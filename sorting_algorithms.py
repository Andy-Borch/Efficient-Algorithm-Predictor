#Implement various sorting algorithms from the data in the csv
#Add run time for each algorithm on each peice of data to the csv file
import time
import random
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt

lengths = [10, 100, 500, 1000]
deviations = [100, 500, 1000]

def merge_sort_helper(arr):
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort_helper(arr[:mid])
    right = merge_sort_helper(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] < right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

def quick_sort_helper(arr):
    if len(arr) <= 1:
        return arr
    pivot = arr[len(arr) // 2]
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]
    return quick_sort_helper(left) + middle + quick_sort_helper(right)

def heapify(arr, n, i):
    largest = i
    left = 2 * i + 1
    right = 2 * i + 2

    if left < n and arr[largest] < arr[left]:
        largest = left

    if right < n and arr[largest] < arr[right]:
        largest = right

    if largest != i:
        arr[i], arr[largest] = arr[largest], arr[i]
        heapify(arr, n, largest)

def counting_sort(arr, exp):
    n = len(arr)
    output = [0] * n
    count = [0] * 10

    for i in range(n):
        index = (arr[i] // exp) % 10
        count[index] += 1

    for i in range(1, 10):
        count[i] += count[i - 1]

    i = n - 1
    while i >= 0:
        index = (arr[i] // exp) % 10
        output[count[index] - 1] = arr[i]
        count[index] -= 1
        i -= 1

    for i in range(n):
        arr[i] = output[i:][0]

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
    for i in range(1, len(data)):
        key = data[i]
        j = i - 1
        while j >= 0 and key < data[j]:
            data[j + 1] = data[j]
            j -= 1
        data[j + 1] = key
    return data

def merge_sort(data):
    if isinstance(data, (tuple, set, pd.Series)):
        data = list(data)
    return merge_sort_helper(data)

def quick_sort(data):
    if isinstance(data, tuple) or isinstance(data, set):
        data = list(data)
    return quick_sort_helper(data)

def heap_sort(data):
    if isinstance(data, tuple) or isinstance(data, set):
        data = list(data)
    n = len(data)
    for i in range(n // 2 - 1, -1, -1):
        heapify(data, n, i)

    for i in range(n - 1, 0, -1):
        data[0], data[i] = data[i], data[0]
        heapify(data, i, 0)
    return data

def radix_sort(data):
    if isinstance(data, tuple) or isinstance(data, set):
        data = list(data)
    if len(data) == 0:
        return data

    max_num = max(data)
    exp = 1
    while max_num // exp > 0:
        counting_sort(data, exp)
        exp *= 10
    return data

def bucket_sort(data):
    #Must convert np array and pd Series to list because they
    #dont have clear and extend methods
    #Thats ok, just another quirk of the data the model will have to predict
    if isinstance(data, (tuple, set, np.ndarray, pd.Series)):
        data = list(data)
    if len(data) == 0:
        return data

    min_value = min(data)
    max_value = max(data)
    bucket_count = len(data)
    buckets = [[] for _ in range(bucket_count)]

    for num in data:
        index = int((num - min_value) * (bucket_count - 1) / (max_value - min_value))
        buckets[index].append(num)

    data.clear()
    for bucket in buckets:
        insertion_sort(bucket)
        data.extend(bucket)
    return data

def generate_data():
    #This data will eventually come from the csv file, this is just proof of concept for now
    length = random.randint(0, 3000)
    deviation = random.randint(0, 100)

    values = np.random.normal(scale=deviation, size=length)

    array = np.round(values).astype(int).tolist()
    random.shuffle(array)
    #TODO: Make each data structure type have different sizes/content to make more realistic to actual data we will use?
    return {
        'Array': array.copy(),
        'Tuple': tuple(array.copy()),
        'Numpy Array': np.array(array.copy()),
        'List': list(array.copy()),
        'Pandas Series': pd.Series(array.copy()),
        'Set': set(array.copy())
    }

def time_sort(algorithm_name, sort_func):
    base_data = generate_data()
    timings = {}

    for label, original in base_data.items():
        if isinstance(original, np.ndarray):
            data = original.copy()
        elif isinstance(original, pd.Series):
            data = original.copy(deep=True)
        elif isinstance(original, (list, set, tuple)):
            data = copy.deepcopy(original)
        else:
            data = list(original)

        start = time.perf_counter()
        try:
            sort_func(data)
        except Exception as e:
            timings[label] = f"Error: {e}"
        else:
            elapsed_time_ms = (time.perf_counter() - start) * 1000
            timings[label] = round(elapsed_time_ms, 5)

    df = pd.DataFrame(list(timings.items()), columns=['Data Structure Type', 'Time (ms)'])
    print(f"\n----- {algorithm_name} SORT RESULTS -----")
    print(df)
    return df


def plot_sorting_times():
    plt.figure(figsize=(16, 12))

    algorithms = {
        'Bubble Sort': bubble_sort,
        'Insertion Sort': insertion_sort,
        'Merge Sort': merge_sort,
        'Quick Sort': quick_sort,
        'Heap Sort': heap_sort,
        'Radix Sort': radix_sort,
        'Bucket Sort': bucket_sort
    }

    for algorithm_name, sort_func in algorithms.items():
        all_timings = []
        all_lengths = []
        all_devs = []
        all_types = []

        for length in lengths:
            for deviation in deviations:
                print(f"Running {algorithm_name} for length={length}, deviation={deviation}")
                data_structures = generate_data()
                timings = time_sort(algorithm_name, sort_func)

                if not isinstance(timings, pd.DataFrame):
                    print(f"Error: timings is not a DataFrame. Received: {timings}")
                    continue

                for _, row in timings.iterrows():
                    all_timings.append(row['Time (ms)'])
                    all_lengths.append(length)
                    all_devs.append(deviation)
                    all_types.append(row['Data Structure Type'])

        if all_timings and all_lengths and all_devs and all_types:
            df = pd.DataFrame({
                'Time (ms)': all_timings,
                'Length': all_lengths,
                'Deviation': all_devs,
                'Data Structure': all_types
            })

            plt.scatter(df['Length'], df['Time (ms)'], label=f'{algorithm_name} (Length)', alpha=0.5)
            plt.scatter(df['Deviation'], df['Time (ms)'], label=f'{algorithm_name} (Deviation)', alpha=0.5)

        else:
            print(f"Skipping plot for {algorithm_name} due to empty timing data.")

    plt.xlabel('Length or Deviation of Data')
    plt.ylabel('Time (ms)')
    plt.title('Sorting Algorithm Performance')
    plt.legend(title="Algorithm", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.show()

print("-----THIS IS ONLY FOR PROOF OF CONCEPT. EACH SORTING FUNCTION IS BEING TESTED ACROSS MULTIPLE DATA STRUCTURES.-----")

#TODO: Print fastest, then how much slower in % the rest are?
#TODO: Make a for loop that changes data characteristics like length, and deviation and plot time vs certain characteristic? with a
#  scatterplot. y axis is time, x axis is a feature like length etc, each color dot represents a data structure type
    #make a scatter plot for each algorithm 


#time_sort("Bubble", bubble_sort)
#time_sort("Insertion", insertion_sort)
#TODO: Get rid of merge sort?
#time_sort("Merge", merge_sort)
#time_sort("Quick", quick_sort)
#time_sort("Heap", heap_sort)
#time_sort("Radix", radix_sort)
#time_sort("Bucket", bucket_sort)

plot_sorting_times()