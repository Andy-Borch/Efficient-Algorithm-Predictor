#Implement various sorting algorithms from the data in the csv
#Add run time for each algorithm on each peice of data to the csv file
import time
import random
import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import os
import seaborn as sns

lengths = [10, 100, 500]
deviations = [10, 100, 500]

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

    size = random.randint(10, 1000)
    std_dev = random.randint(1, 100)

    values = np.random.normal(loc=0, scale=std_dev, size=size)
    values = np.round(values).astype(int).tolist()
    random.shuffle(values)

    array = values.copy()
    data_structures = {
        'Array': array,
        'List': list(array),
        'Tuple': tuple(array),
        'Set': set(array),
        'NPArray': np.array(array),
        'PandasSeries': pd.Series(array)
    }

    num_unique_elements = len(set(array))

    return {
        'Data Structures': data_structures,
        'Size': size,
        'StandardDeviation': std_dev,
        'NumUniqueElements': num_unique_elements
    }

def time_sort(algorithm_name, sort_func):
    base_data = generate_data()
    data_structures = base_data['Data Structures']
    num_unique_elements = base_data['NumUniqueElements']
    timings = {}

    for label, original in data_structures.items():
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
    df['NumUniqueElements'] = num_unique_elements
    print(f"\n----- {algorithm_name} SORT RESULTS -----")
    print(df)
    return df

def plot_sorting_times():
    # Ensure the output directory exists
    output_dir = "plots"
    os.makedirs(output_dir, exist_ok=True)

    algorithms = {
        'Bubble Sort': bubble_sort,
        'Insertion Sort': insertion_sort,
        'Merge Sort': merge_sort,
        'Quick Sort': quick_sort,
        'Heap Sort': heap_sort,
        'Radix Sort': radix_sort,
        'Bucket Sort': bucket_sort
    }

    # Define a consistent color palette for data structures
    data_structures = ['Array', 'List', 'Tuple', 'Set', 'NPArray', 'PandasSeries']
    ds_colors = sns.color_palette("tab10", len(data_structures))
    ds_color_map = {ds: ds_colors[i] for i, ds in enumerate(data_structures)}

    all_data = []

    # Collect data for all algorithms
    for algorithm_name, sort_func in algorithms.items():
        for length in lengths:
            for deviation in deviations:
                print(f"Running {algorithm_name} for length={length}, deviation={deviation}")
                base_data = generate_data()
                data_structures = base_data['Data Structures']
                num_unique_elements = base_data['NumUniqueElements']
                timings = time_sort(algorithm_name, sort_func)

                if not isinstance(timings, pd.DataFrame):
                    print(f"Error: timings is not a DataFrame. Received: {timings}")
                    continue

                for _, row in timings.iterrows():
                    all_data.append({
                        'Algorithm': algorithm_name,
                        'Data Structure': row['Data Structure Type'],
                        'Time (ms)': row['Time (ms)'],
                        'Length': length,
                        'Deviation': deviation,
                        'NumUniqueElements': num_unique_elements
                    })

    if all_data:
        df = pd.DataFrame(all_data)
        df = df[df['Time (ms)'] <= df['Time (ms)'].quantile(0.95)]

        # Features to plot
        features = ['Length', 'Deviation', 'NumUniqueElements']

        # Create a plot for each algorithm and feature
        for algorithm_name in df['Algorithm'].unique():
            for feature in features:
                plt.figure(figsize=(12, 8))
                subset = df[df['Algorithm'] == algorithm_name]

                # Plot each data structure type as a separate series
                for data_structure in subset['Data Structure'].unique():
                    ds_subset = subset[subset['Data Structure'] == data_structure]
                    plt.scatter(
                        ds_subset[feature],
                        ds_subset['Time (ms)'],
                        label=data_structure,
                        alpha=0.8,
                        color=ds_color_map[data_structure],
                        edgecolor='black'
                    )

                    if len(ds_subset) > 1:
                        x = ds_subset[feature]
                        y = ds_subset['Time (ms)']
                        coeffs = np.polyfit(x, y, 1)
                        best_fit_line = np.poly1d(coeffs)

                        # Plot the best-fit line
                        x_range = np.linspace(x.min(), x.max(), 100)
                        plt.plot(
                            x_range,
                            best_fit_line(x_range),
                            color=ds_color_map[data_structure],
                            linestyle='--',
                            label=f'{data_structure} (Best Fit)'
                        )

                plt.xlabel(feature, fontsize=14)
                plt.ylabel('Time (ms)', fontsize=14)
                plt.yscale('log')
                plt.title(f'{algorithm_name} - Runtime vs {feature}', fontsize=16)
                plt.legend(title="Data Structure", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=12)
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.tight_layout()

                # Save the plot to a file
                filename = f"{algorithm_name.replace(' ', '_')}_vs_{feature}.png"
                filepath = os.path.join(output_dir, filename)
                plt.savefig(filepath, dpi=300)
                print(f"Saved plot: {filepath}")
                plt.close()
    else:
        print("No data to plot.")

print("-----THIS IS ONLY FOR PROOF OF CONCEPT. EACH SORTING FUNCTION IS BEING TESTED ACROSS MULTIPLE DATA STRUCTURES.-----")

plot_sorting_times()
print("Plotting completed.")
