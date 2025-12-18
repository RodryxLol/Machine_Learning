import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib

ARTIFACTS_PATH = "artifacts/"

def main():

    df_scaled = pd.read_csv(ARTIFACTS_PATH + "df_scaled.csv") #se cargan los datos escalados
    X_train = pd.read_csv(ARTIFACTS_PATH + "X_train_scaled.csv")
  
    K_range = range(2, 11) #Se define un rango de valores posibles para el número de clusters
    inertia_values = []

    for k in K_range: #Entrenamiento para el metodo del codo
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )
        kmeans.fit(X_train)
        inertia_values.append(kmeans.inertia_)


    plt.figure(figsize=(7, 4)) #Se dibuja el gráfico del método del codo
    plt.plot(K_range, inertia_values, marker="o")
    plt.title("Método del Codo (Training Set)")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inercia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    k_optimo = 4 #se define el numero final de grupos basado en el grafico de codo

    kmeans_final = KMeans( #Se entrena el modelo K-Means definitivo usando el valor óptimo
        n_clusters=k_optimo,
        random_state=42,
        n_init=20
    )
    kmeans_final.fit(X_train)

    joblib.dump(
        kmeans_final,
        os.path.join(ARTIFACTS_PATH, "kmeans_model.pkl")
    )

    print("Modelo K-Means entrenado correctamente (sin data leakage).")

if __name__ == "__main__":
    main()
