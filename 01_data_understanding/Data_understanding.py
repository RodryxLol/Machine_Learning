import pandas as pd

DATA_PATH = "data/application_train.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    print("Se muestra el numero de filas y clumnas del dataset", df.shape)
    print("\nInformación sobre los datos:")
    print(df.info())

    print("\nSe identifican los valores faltantes:")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nResumen de estadisticas:")
    print(df.describe())

    duplicated = df.duplicated().sum()
    print(f"\nRegistro duplicados: {duplicated}")

    print("\nColumnas del dataset:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()
