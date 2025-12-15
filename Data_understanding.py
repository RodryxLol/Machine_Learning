import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import pca

from google.colab import files
uploaded = files.upload()
for fn in uploaded.keys():
    name=fn
df = pd.read_csv(name, sep=",", na_values=['', ' ', 'NA', 'N/A', 'null'])

df.head()

df.info()

nombres_columna = list(df.columns)
print("\nNombres de las columnas:")
print(nombres_columna)

df.isna().sum()

df.nunique()

df_dup = df.duplicated(keep=False)
fila_dup = df[df_dup]
print(fila_dup)

df.describe()

df["DAYS_BIRTH"]

df["DAYS_EMPLOYED"]

