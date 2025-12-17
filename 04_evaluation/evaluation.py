import pandas as pd
import joblib
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

ARTIFACTS_PATH = "artifacts/"


def main():

    X_train = pd.read_csv(ARTIFACTS_PATH + "X_train_scaled.csv")
    X_test = pd.read_csv(ARTIFACTS_PATH + "X_test_scaled.csv")

    df_train = pd.read_csv(ARTIFACTS_PATH + "df_train_variables.csv")
    df_test = pd.read_csv(ARTIFACTS_PATH + "df_test_variables.csv")

    kmeans = joblib.load(ARTIFACTS_PATH + "kmeans_model.pkl")


    df_train["CLUSTER"] = kmeans.predict(X_train)
    df_test["CLUSTER"] = kmeans.predict(X_test)


    if "TARGET" in df_train.columns:
        print("\nTasa de morosidad por cluster (TRAIN):")
        print(df_train.groupby("CLUSTER")["TARGET"].mean())

    if "TARGET" in df_test.columns:
        print("\nTasa de morosidad por cluster (TEST):")
        print(df_test.groupby("CLUSTER")["TARGET"].mean())


    pca = PCA(n_components=2, random_state=42)
    X_train_pca = pca.fit_transform(X_train)
    X_test_pca = pca.transform(X_test)


    df_train_pca = pd.DataFrame(X_train_pca, columns=["PC1", "PC2"])
    df_train_pca["CLUSTER"] = df_train["CLUSTER"].values
    df_train_pca["SET"] = "Train"

    df_test_pca = pd.DataFrame(X_test_pca, columns=["PC1", "PC2"])
    df_test_pca["CLUSTER"] = df_test["CLUSTER"].values
    df_test_pca["SET"] = "Test"

    df_pca = pd.concat([df_train_pca, df_test_pca])


    plt.figure(figsize=(8, 6))

    for c in sorted(df_pca["CLUSTER"].unique()):
        subset = df_pca[df_pca["CLUSTER"] == c]
        plt.scatter(
            subset["PC1"],
            subset["PC2"],
            alpha=0.5,
            label=f"Cluster {c}"
        )

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Clusters K-Means visualizados con PCA (Train + Test)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    print("\nVarianza explicada por PC1 y PC2:", pca.explained_variance_ratio_)


if __name__ == "__main__":
    main()
