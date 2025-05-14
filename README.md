# Machine Learning Fastest-Algorithm-Predictor
## CS345 Machine Learning Foundationa Final Project
### Authors: Andy Borch and Xander Gupton

---

## Overview

This project investigates how various characteristics of data collections affect the efficiency of different sorting algorithms. It further explores whether sorting performance can be accurately predicted using machine learning models based on those features.

## Goals

- Generate synthetic datasets with controlled statistical features.

- Benchmark multiple sorting algorithms on these datasets.

- Train ML models to predict sorting algorithm runtime from dataset characteristics.

## Features

   - **Synthetic Data Generation:** Custom arrays are generated with adjustable features like size, mean, standard deviation, and number of unique elements.

   - **Data Types:** Arrays are stored using different Python/Numpy types (list, tuple, set ndarray) with binary one-hot encoding.

   - **Sorting Algorithms:** Includes Quick Sort, Heap Sort, Bubble Sort, Insertion Sort, Radix Sort, and Bucket Sort. Algorithms are adapted to handle various data types.

   - **Runtime Measurement:** Execution time is measured and appended as target values for ML prediction.

   - **Machine Learning Models:** Several classifiers (e.g., KNN, Random Forest, SVM, Neural Networks) are trained to predict the fastest sorting algorithm based on features.
