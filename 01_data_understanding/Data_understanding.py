import pandas as pd

DATA_PATH = "data/application_train.csv"

def main():
    df = pd.read_csv(DATA_PATH)

    print("Dimensiones del dataset:", df.shape)
    print("\nInformación general:")
    print(df.info())

    print("\nValores nulos por columna:")
    print(df.isna().sum().sort_values(ascending=False).head(20))

    print("\nDescripción estadística:")
    print(df.describe())

    duplicated = df.duplicated().sum()
    print(f"\nFilas duplicadas: {duplicated}")

    print("\nColumnas disponibles:")
    print(df.columns.tolist())

if __name__ == "__main__":
    main()
