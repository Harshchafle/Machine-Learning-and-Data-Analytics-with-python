import numpy as np
import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 22],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston'],
    'Salary': [50000, 75000, 80000, 65000, 55000]
}

df = pd.DataFrame(data)
print("Data Info:")
print(df.info())
print("\n")
print("DataFrame Description:")
print(df.describe())
print("\n")
print("Median of DataFrame Description:")
print(df.describe().median())
print("\n")
print("DataFrame from list of lists:")
data_list = [
    ['Eve', 22, 'Boston'],
    ['Frank', 27, 'Miami'],
    ['Grace', 33, 'Seattle']
]
df2 = pd.DataFrame(data_list, columns=['Name', 'Age', 'City'])
print(df2)
print("\n")


df.to_csv('dataframe_output.csv', index=False)
print("DataFrame saved to 'dataframe_output.csv'")