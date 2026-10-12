#Week-1 
#Day -1


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


#Day-2

# PANDAS FOR DATA MANIPULATION
# Stack: NumPy, Pandas, Scikit-Learn


import os
import time
import numpy as np
import pandas as pd

# Install Parquet support if required
try:
    import pyarrow
except ImportError:
    import subprocess
    subprocess.check_call(["python", "-m", "pip", "install", "pyarrow"])
    import pyarrow

# 1. LOAD A REAL INDIAN DATASET
# Indian Air Quality dataset (public GitHub CSV)
url = "https://raw.githubusercontent.com/plotly/datasets/master/india-air-quality-data.csv"

try:
    df = pd.read_csv(url)
    print("Real Indian dataset loaded successfully.")
except Exception:
    print("Online dataset unavailable.")
    print("Upload an Indian dataset CSV named 'indian_dataset.csv'.")

    # Try a CSV uploaded to the current working directory.
    if os.path.exists("indian_dataset.csv"):
        df = pd.read_csv("indian_dataset.csv")
    else:
        # Fallback for practising only; this is synthetic data.
        df = pd.DataFrame({
            "City": ["Bengaluru", "Chennai", "Mumbai", "Bengaluru",
                     "Chennai", "Mumbai", "Delhi", "Delhi"],
            "State": ["Karnataka", "Tamil Nadu", "Maharashtra",
                      "Karnataka", "Tamil Nadu", "Maharashtra",
                      "Delhi", "Delhi"],
            "Year": [2023, 2023, 2023, 2024, 2024, 2024, 2023, 2024],
            "PM25": [32, 40, 55, 28, 38, 50, 90, 85],
            "PM10": [60, 70, 85, 55, 65, 80, 120, 115]
        })

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Data Types:")
print(df.dtypes)

print("\nFirst 10 Rows:")
print(df.head(10))

# 2. DATA CLEANING
# Remove duplicate records.
df = df.drop_duplicates()

# Replace infinite values with missing values.
df = df.replace([np.inf, -np.inf], np.nan)

# Fill missing numerical values with their column median.
numeric_cols = df.select_dtypes(include=np.number).columns
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with the most frequent value.
categorical_cols = df.select_dtypes(include=["object", "category"]).columns
for col in categorical_cols:
    if df[col].notna().any():
        df[col] = df[col].fillna(df[col].mode()[0])

print("\nCleaned Dataset Shape:")
print(df.shape)

# 3. FILTER OPERATION
# Filter rows where at least one numeric column is above its median.
if len(numeric_cols) > 0:
    filter_col = numeric_cols[0]
    filtered_df = df[df[filter_col] > df[filter_col].median()]

    print(f"\nFiltered rows: {filter_col} above its median")
    print(filtered_df.head(10))
else:
    filtered_df = df.copy()
    print("\nNo numeric columns available for filtering.")


# 4. GROUPBY OPERATION
# Group by the first categorical column and calculate numeric means.
if len(categorical_cols) > 0 and len(numeric_cols) > 0:
    group_col = categorical_cols[0]
    grouped_df = df.groupby(group_col)[list(numeric_cols)].mean()

    print(f"\nGroupby: Mean numerical values by {group_col}")
    print(grouped_df)
else:
    grouped_df = pd.DataFrame()
    print("\nSuitable columns for groupby are unavailable.")


# 5. MERGE OPERATION
# Create a small lookup table and merge it using a common column.
if len(categorical_cols) > 0:
    merge_col = categorical_cols[0]

    lookup_df = df[[merge_col]].drop_duplicates().copy()
    lookup_df["Record_ID"] = range(1, len(lookup_df) + 1)

    merged_df = pd.merge(
        df,
        lookup_df,
        on=merge_col,
        how="left"
    )

    print(f"\nMerged DataFrame using {merge_col}:")
    print(merged_df.head(10))
else:
    merged_df = df.copy()
    print("\nNo categorical column available for merging.")


# 6. PIVOT_TABLE OPERATION
# Summarise the first numeric column by the first categorical column.
if len(categorical_cols) > 0 and len(numeric_cols) > 0:
    pivot_df = pd.pivot_table(
        df,
        index=categorical_cols[0],
        values=numeric_cols[0],
        aggfunc="mean"
    )

    print("\nPivot Table:")
    print(pivot_df)
else:
    pivot_df = pd.DataFrame()
    print("\nSuitable columns for pivot_table are unavailable.")


# 7. EXPORT CLEANED DATA TO CSV AND PARQUET
# Export the cleaned DataFrame to CSV format.
csv_file = "cleaned_indian_data.csv"
df.to_csv(csv_file, index=False)

# Export the same DataFrame to Parquet format.
parquet_file = "cleaned_indian_data.parquet"
df.to_parquet(parquet_file, index=False)

print("\nFiles exported successfully.")

# Compare file sizes in KB.
csv_size = os.path.getsize(csv_file) / 1024
parquet_size = os.path.getsize(parquet_file) / 1024

print(f"CSV file size: {csv_size:.2f} KB")
print(f"Parquet file size: {parquet_size:.2f} KB")

if csv_size > 0:
    reduction = (csv_size - parquet_size) / csv_size * 100
    print(f"Parquet size reduction compared with CSV: {reduction:.2f}%")

if parquet_size < csv_size:
    print("Parquet file is smaller for this dataset.")
elif parquet_size > csv_size:
    print("CSV file is smaller for this dataset.")
else:
    print("Both files have the same size.")

# 8. VERIFY EXPORTED FILES
print("\nCSV file preview:")
print(pd.read_csv(csv_file).head())

print("\nParquet file preview:")
print(pd.read_parquet(parquet_file).head())

print("\nAll practical operations completed.")



#Day-3
# Create a sample Indian dataset
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

df = pd.DataFrame({
    "City": ["Bengaluru", "Chennai", "Mumbai", "Delhi", "Bengaluru", "Mumbai"],
    "State": ["Karnataka", "Tamil Nadu", "Maharashtra", "Delhi", "Karnataka", "Maharashtra"],
    "Population_Lakhs": [140, 100, np.nan, 200, 140, 125],
    "AQI": [80, 65, 120, np.nan, 80, 110]
})

# Inspect dataset
print("Original Shape:", df.shape)
print(df.head())

# Clean data
df = df.replace([np.inf, -np.inf], np.nan)
df = df.drop_duplicates()
df = df.dropna(axis=1, how="all")

# Fill missing numerical and categorical values
num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(exclude=np.number).columns

if len(num_cols):
    df[num_cols] = SimpleImputer(strategy="median").fit_transform(df[num_cols])

if len(cat_cols):
    df[cat_cols] = SimpleImputer(strategy="most_frequent").fit_transform(df[cat_cols])

# Test the cleaned data
assert not df.duplicated().any()
assert not df.isnull().any().any()

print("\nCleaned Shape:", df.shape)
print("\nMissing Values:\n", df.isnull().sum())
print("\nCleaned Dataset:\n", df)

# Export cleaned dataset
df.to_csv("cleaned_dataset.csv", index=False)
print("\nSuccess! cleaned_dataset.csv saved.")



# Day 4: Exploratory Data Analysis (EDA)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load dataset (using the Day 3 cleaned file)
df = pd.read_csv("cleaned_dataset.csv")

# 2. Inspect dataset
print("Dataset Description:\n", df.describe())
print("\nDataset Information:")
df.info()
print("\nMissing Values:\n", df.isnull().sum())

# 3. Plot distributions of numeric columns
df.hist(figsize=(10, 6), bins=10)
plt.tight_layout()
plt.show()

# 4. Correlation heatmap
numeric_df = df.select_dtypes(include=np.number)
if numeric_df.shape[1] >= 2:
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.show()
else:
    print("At least two numeric columns are needed for a correlation heatmap.")

# 5. Top 10 category counts
cat_cols = df.select_dtypes(exclude=np.number).columns
if len(cat_cols):
    for col in cat_cols:
        print(f"\nTop 10 categories in {col}:")
        print(df[col].value_counts().head(10))
        df[col].value_counts().head(10).plot(kind="bar", title=f"Top Categories: {col}")
        plt.tight_layout()
        plt.show()

# 6. Five observations
print("""
EDA Observations:
1. Describe shows the average and spread of numeric values.
2. Histograms show the distribution of numeric columns.
3. The heatmap shows relationships between numeric variables.
4. Category counts show the most frequent groups.
5. Missing values and unusual values need further checking.
""")

# 7. Save EDA output
print("\nEDA completed successfully!")




# Day 5: Data Visualisation using Matplotlib and Seaborn

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load the cleaned dataset
df = pd.read_csv("cleaned_dataset.csv")

# 2. Bar chart - average values by category
cat_cols = df.select_dtypes(exclude="number").columns
num_cols = df.select_dtypes(include="number").columns

if len(cat_cols) and len(num_cols):
    summary = df.groupby(cat_cols[0])[num_cols[0]].mean()
    summary.plot(kind="bar", title="Average by Category")
    plt.ylabel(num_cols[0])
    plt.tight_layout()
    plt.show()

# 3. Histogram - distribution of numerical data
if len(num_cols):
    df[num_cols[0]].plot(kind="hist", bins=10, title="Data Distribution")
    plt.xlabel(num_cols[0])
    plt.tight_layout()
    plt.show()

# 4. Scatter plot - relationship between two numeric columns
if len(num_cols) >= 2:
    sns.scatterplot(data=df, x=num_cols[0], y=num_cols[1])
    plt.title("Relationship Between Variables")
    plt.tight_layout()
    plt.show()

# 5. Heatmap - correlation between numerical columns
if len(num_cols) >= 2:
    sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
    plt.title("Correlation Heatmap")
    plt.tight_layout()
    plt.show()

# 6. Test and finish
assert not df.empty, "Dataset is empty!"
print("Visualisation completed successfully!")
print("Rows:", len(df), "| Columns:", len(df.columns))


