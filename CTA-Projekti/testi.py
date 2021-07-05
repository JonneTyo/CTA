import pandas as pd
from itertools import product
from sklearn.utils.extmath import cartesian
import numpy as np

def my_generator():
    for i in range(1, 10):
        yield i

x = my_generator()
print(next(x))
for i in x:
    print(i)
