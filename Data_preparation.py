#Se cambia el formato de las columnas de days_birth y days_employed para un mejor entendimiento (en años).

df['AGE'] = df['DAYS_BIRTH'] / 365
df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / 365

df.loc[df['DAYS_EMPLOYED'] == 365243, 'YEARS_EMPLOYED'] = np.nan
df.loc[df['YEARS_EMPLOYED'] > 60, 'YEARS_EMPLOYED'] = np.nan

df['YEARS_EMPLOYED'] = df['YEARS_EMPLOYED'].fillna(df['YEARS_EMPLOYED'].median())

#Se hace una selección de variables que se consideran útiles a la hora de hacer el clustering (agregar más si se considera necesario)

VARIABLES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AGE",
    "YEARS_EMPLOYED"
]

df_variables = df[VARIABLES].copy()
df_variables.head()

df_variables.describe()

#Se hace imputación con la mediana
df['YEARS_EMPLOYED'] = df['YEARS_EMPLOYED'].fillna(df['YEARS_EMPLOYED'].median())

for col in VARIABLES:
    median_val = df_variables[col].median()
    df_variables[col] = df_variables[col].fillna(median_val)

#Escalamiento estándar
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_variables)

df_scaled = pd.DataFrame(X_scaled, columns=VARIABLES)
df_scaled.head()