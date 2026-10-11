import numpy as np

print("NumPy Fundamentals - AI/ML Internship")
print("NumPy imported successfully!")

# Create a 1D NumPy array
arr1 = np.array([10, 20, 30, 40, 50])

print("1D Array:", arr1)
print("Shape:", arr1.shape)
print("Dimensions:", arr1.ndim)

# Create a 2D NumPy array
arr2 = np.array([[10, 20, 30],
                 [40, 50, 60]])

print("\n2D Array:")
print(arr2)

print("Shape:", arr2.shape)
print("Dimensions:", arr2.ndim)

# Create a 3D NumPy array
arr3 = np.array([
    [[1, 2, 3],
     [4, 5, 6]],

    [[7, 8, 9],
     [10, 11, 12]]
])

print("\n3D Array:")
print(arr3)

print("Shape:", arr3.shape)
print("Dimensions:", arr3.ndim)

# Vectorized Operations
numbers = np.array([10, 20, 30, 40, 50])

print("\nOriginal Array:", numbers)
print("Addition:", numbers + 5)
print("Subtraction:", numbers - 5)
print("Multiplication:", numbers * 2)
print("Division:", numbers / 10)

import numpy as np

numbers = np.array([10, 20, 30, 40, 50])

result = numbers + 10

print("Original Array:", numbers)
print("After Broadcasting:", result)

# Broadcasting with a 2D array
matrix = np.array([[1, 2, 3],
                   [4, 5, 6]])

result = matrix + 10

print("\nOriginal Matrix:")
print(matrix)

print("After Broadcasting:")
print(result)


# Matrix Multiplication using NumPy

import numpy as np

matrix_a = np.array([
    [1, 2],
    [3, 4]
])

matrix_b = np.array([
    [5, 6],
    [7, 8]
])

# Multiply the matrices without Python loops
result = np.matmul(matrix_a, matrix_b)

print("Matrix A:")
print(matrix_a)

print("\nMatrix B:")
print(matrix_b)

print("\nMatrix Multiplication Result:")
print(result)


# Load CSV dataset using NumPy

data = np.loadtxt(
    "W1D1_NumPy_Fundamentals/student_performance.csv",
    delimiter=",",
    skiprows=1
)

print("\nCSV Dataset:")
print(data)

print("\nDataset Shape:", data.shape)

# Calculate Mean

study_hours = data[:, 1]
attendance = data[:, 2]
marks = data[:, 3]

print("\nMean Study Hours:", np.mean(study_hours))
print("Mean Attendance:", np.mean(attendance))
print("Mean Marks:", np.mean(marks))


# Calculate Standard Deviation

print("\nStandard Deviation:")

print("Study Hours:", np.std(study_hours))
print("Attendance:", np.std(attendance))
print("Marks:", np.std(marks))


# Calculate Correlation

correlation = np.corrcoef(study_hours, marks)

print("\nCorrelation between Study Hours and Marks:")
print(correlation)

print("\nCorrelation Coefficient:", correlation[0, 1])


# Load the real Iris dataset

# Load the Iris dataset

iris_data = np.loadtxt(
    "iris.data",
    delimiter=",",
    usecols=(0, 1, 2, 3)
)

print("\nFirst 5 rows of Iris Dataset:")
print(iris_data[:5])

print("\nDataset Shape:", iris_data.shape)

# Calculate statistics using the Iris dataset

print("\nIris Dataset Statistics")

# Mean of each column
mean_values = np.mean(iris_data, axis=0)
print("Mean:", mean_values)

# Standard deviation of each column
std_values = np.std(iris_data, axis=0)
print("Standard Deviation:", std_values)


# Calculate correlation between Iris flower measurements

correlation = np.corrcoef(iris_data, rowvar=False)

print("\nCorrelation Matrix:")
print(correlation)
