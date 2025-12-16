import pandas as pd
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ARTIFACTS_PATH = "artifacts/"

def main():

    df_scaled = pd.read_csv(ARTIFACTS_PATH + "df_scaled.csv")
    kmeans = joblib.load(ARTIFACTS_PATH + "kmeans_model.pkl")


    df_scaled["CLUSTER"] = kmeans.predict(df_scaled)
  
    if "TARGET" in df_scaled.columns:
        tasa_morosidad = df_scaled.groupby("CLUSTER")["TARGET"].mean()
        print("Tasa de morosidad por cluster:")
        print(tasa_morosidad)

    pca = PCA(n_components=2, random_state=42)
    X_pca = pca.fit_transform(df_scaled.drop(columns="CLUSTER"))

    df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
    df_pca["CLUSTER"] = df_scaled["CLUSTER"]


    plt.figure(figsize=(8,6))
    for c in sorted(df_pca["CLUSTER"].unique()):
        subset = df_pca[df_pca["CLUSTER"] == c]
        plt.scatter(subset["PC1"], subset["PC2"], alpha=0.6, label=f"Cluster {c}")

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters visualizados en espacio PCA")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
