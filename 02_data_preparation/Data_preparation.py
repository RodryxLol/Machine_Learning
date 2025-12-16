import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib

DATA_PATH = "data/application_train.csv"
ARTIFACTS_PATH = "artifacts/"
df = pd.read_csv(DATA_PATH)

def clip_outliers(df, cols, lower=0.01, upper=0.99):
    df = df.copy()
    for col in cols:
        q_low = df[col].quantile(lower)
        q_high = df[col].quantile(upper)
        df[col] = df[col].clip(q_low, q_high)
    return df


def main():
    df['AGE'] = df['DAYS_BIRTH'] / 365
    df['YEARS_EMPLOYED'] = df['DAYS_EMPLOYED'] / 365

    df.loc[df['DAYS_EMPLOYED'] == 365243, 'YEARS_EMPLOYED'] = np.nan
    df.loc[df['YEARS_EMPLOYED'] > 60, 'YEARS_EMPLOYED'] = np.nan

    df['YEARS_EMPLOYED'] = df['YEARS_EMPLOYED'].fillna(df['YEARS_EMPLOYED'].median())

        
    VARIABLES = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AGE",
        "YEARS_EMPLOYED"
    ]

    df_variables = df[VARIABLES].copy()
    df_variables.head()

    
    df['YEARS_EMPLOYED'] = df['YEARS_EMPLOYED'].fillna(df['YEARS_EMPLOYED'].median())

    for col in VARIABLES:
        median_val = df_variables[col].median()
        df_variables[col] = df_variables[col].fillna(median_val)

    df_variables = clip_outliers(df_variables, VARIABLES)
    
    df_variables.to_csv(ARTIFACTS_PATH + "df_variables.csv", index=False)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_variables)
    df_scaled = pd.DataFrame(X_scaled, columns=VARIABLES)
    
    
    joblib.dump(scaler, ARTIFACTS_PATH + "scaler.pkl")

    df_scaled = pd.DataFrame(X_scaled, columns=VARIABLES)
    df_scaled.to_csv(ARTIFACTS_PATH + "df_scaled.csv", index=False)
    df_variables.to_csv(ARTIFACTS_PATH + "df_variables.csv", index=False)
    df_scaled.head()
    

    

if __name__ == "__main__":
    main()
