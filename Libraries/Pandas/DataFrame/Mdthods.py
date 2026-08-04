import pandas as pd

data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David', 'Eve'],
    'Age': [25, 30, 35, 28, 22],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago', 'Boston'],
    'Salary': [50000, 75000, 80000, 65000, 55000]
}
df = pd.DataFrame(data)
#print(df)

print("Data Info  ")
print(df.info())
print("\n")

print("Dataframe Description : ")
print(df.describe())
print("\n")
print(df.describe().median())