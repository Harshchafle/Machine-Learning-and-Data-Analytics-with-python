import numpy as np
import pandas as pd

data = {
    "a" : [1,2,3,4,5],
    "b" : [2,3,4,5,6],
    "c" : [3,4,5,6,7],
    "d" : [4,5,6,7,8],
    "e" : [5,6,7,8,9]
}
df = pd.DataFrame(data)
#print(df)

#print(df.loc[:2])
#print(df.head(3))

data_list = [
    ['Eve', 22, 'Boston'],
    ['Frank', 27, 'Miami'],
    ['Grace', 33, 'Seattle']
]

df2 = pd.DataFrame(data_list, columns=['Name', 'Age', 'City'])
print("DataFrame from list of lists:")
print(df2)
print("\n")