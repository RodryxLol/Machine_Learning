import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
import joblib

ARTIFACTS_PATH = "artifacts/"

def main():

    df_scaled = pd.read_csv(ARTIFACTS_PATH + "df_scaled.csv")
    X_train = pd.read_csv(ARTIFACTS_PATH + "X_train_scaled.csv")
  
    K_range = range(2, 11)
    inertia_values = []

    for k in K_range:
        kmeans = KMeans(
            n_clusters=k,
            random_state=42,
            n_init=20
        )
        kmeans.fit(X_train)
        inertia_values.append(kmeans.inertia_)


    plt.figure(figsize=(7, 4))
    plt.plot(K_range, inertia_values, marker="o")
    plt.title("Método del Codo (Training Set)")
    plt.xlabel("Número de Clusters (k)")
    plt.ylabel("Inercia")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


    k_optimo = 4

    kmeans_final = KMeans(
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
