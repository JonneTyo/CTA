import pandas as pd

df = pd.DataFrame({'A': [0, 1], 'B': [5, 7]})
id = getattr(df, 'A')
filter1 = id.isin([1, 2, 3, 4])
filter2 = id.isin([0, 2, 3, 4])
print(id.isin(id) & id > 0)
print(df.loc[filter1, :])
