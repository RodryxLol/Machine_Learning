import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
from sklearn.model_selection import train_test_split

DATA_PATH = "data/application_train.csv"
ARTIFACTS_PATH = "artifacts/"
df = pd.read_csv(DATA_PATH)

def clip_outliers_train_test(train, test, cols, lower=0.01, upper=0.99):
    
    #Se limitan los valores extremos basado en percentiles del 
    #train con el objetivo de evitar el data leakage
    
    train = train.copy()
    test = test.copy()

    for col in cols:
        q_low = train[col].quantile(lower) # Percentil Inferior
        q_high = train[col].quantile(upper) # Percentil Superior

        train[col] = train[col].clip(q_low, q_high)
        test[col] = test[col].clip(q_low, q_high)

    return train, test



def main():
    df["AGE"] = df["DAYS_BIRTH"] / -365 #se convierten los dias a años
    df["YEARS_EMPLOYED"] = df["DAYS_EMPLOYED"] / 365 

    df.loc[df["DAYS_EMPLOYED"] == 365243, "YEARS_EMPLOYED"] = np.nan #se eliminan valores irreales
    df.loc[df["YEARS_EMPLOYED"] > 60, "YEARS_EMPLOYED"] = np.nan

    df['YEARS_EMPLOYED'] = df['YEARS_EMPLOYED'].fillna(df['YEARS_EMPLOYED'].median()) #Los valores nulos se reemplazan por la mediana

        
    VARIABLES = [
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AGE",
        "YEARS_EMPLOYED"
    ]


    df_variables = df[VARIABLES].copy()
    df_variables.head()

    df_train, df_test = train_test_split(
        df_variables,
        test_size=0.2,
        random_state=42
    )
    
    df['YEARS_EMPLOYED'] = df['YEARS_EMPLOYED'].fillna(df['YEARS_EMPLOYED'].median())

    for col in VARIABLES:
        median_val = df_train[col].median()
        df_train[col] = df_train[col].fillna(median_val)
        df_test[col] = df_test[col].fillna(median_val)

    df_train, df_test = clip_outliers_train_test(df_train, df_test, VARIABLES) #se aplican limites a los valores extremos
    
    df_variables.to_csv(ARTIFACTS_PATH + "df_variables.csv", index=False)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(df_train)
    X_test_scaled = scaler.transform(df_test)
    X_scaled = scaler.fit_transform(df_variables)
    df_scaled = pd.DataFrame(X_scaled, columns=VARIABLES)
    
    
    joblib.dump(scaler, ARTIFACTS_PATH + "scaler.pkl")

    df_scaled = pd.DataFrame(X_scaled, columns=VARIABLES)
    df_scaled.to_csv(ARTIFACTS_PATH + "df_scaled.csv", index=False)
    df_variables.to_csv(ARTIFACTS_PATH + "df_variables.csv", index=False) #se exportan los datos
    df_scaled.head()

    
    pd.DataFrame(
        X_train_scaled,
        columns=VARIABLES
    ).to_csv(ARTIFACTS_PATH + "X_train_scaled.csv", index=False)

    pd.DataFrame(
        X_test_scaled,
        columns=VARIABLES
    ).to_csv(ARTIFACTS_PATH + "X_test_scaled.csv", index=False)

    df_train.to_csv(ARTIFACTS_PATH + "df_train_variables.csv", index=False)
    df_test.to_csv(ARTIFACTS_PATH + "df_test_variables.csv", index=False)

    print("los datos fueron preparados correctamente")    

    

if __name__ == "__main__":
    main()
