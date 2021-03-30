import pandas as pd

df = pd.DataFrame({'A': [0, 1], 'B': [5, 7]})
id = getattr(df, 'A')
filter1 = id.isin([1, 2, 3, 4])
filter2 = id.isin([])
print(filter1)
print(filter2)
