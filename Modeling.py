
K_range = range(2, 11)
inertia_values = []


for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df_scaled)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(7,4))
plt.plot(K_range, inertia_values, marker='o')
plt.title("Método del Codo")
plt.xlabel("Número de Clusters K")
plt.ylabel("Inercia")
plt.grid(True)
plt.show()

#en este codigo hace el entrenamiento final del modelo K-means
k_optimo = 4

kmeans_final = KMeans(n_clusters=k_optimo, random_state=42)
df_variables["CLUSTER"] = kmeans_final.fit_predict(df_scaled)

df_variables.head()

cluster_summary = df_variables.groupby("CLUSTER").agg({
    "AMT_INCOME_TOTAL": ["mean", "median"],
    "AMT_CREDIT": ["mean", "median"],
    "AMT_ANNUITY": ["mean", "median"],
    "AGE": ["mean", "median"],
    "YEARS_EMPLOYED": ["mean", "median"]
})

cluster_summary

kmeans = KMeans(n_clusters=4, random_state=42)
clusters = kmeans.fit_predict(df_scaled)
df["CLUSTER"] = kmeans.labels_
df["CLUSTER"].nunique()
if "TARGET" in df.columns:
    tasa_morosidad = df.groupby("CLUSTER")["TARGET"].mean().sort_index()
    print("Tasa de morosidad por cluster:")
    print(tasa_morosidad)

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X_scaled)




df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
df_pca["CLUSTER"] = df_variables["CLUSTER"].values



for c in sorted(df_pca["CLUSTER"].unique()):
    subset = df_pca[df_pca["CLUSTER"] == c]
    plt.scatter(subset["PC1"], subset["PC2"], alpha=0.6, label=f"Cluster {c}")



plt.xlabel("PC1")
plt.ylabel("PC2")
plt.title("Clusters K-Means visualizados en espacio PCA")
plt.legend()
plt.show()

print("Varianza explicada por PC1 y PC2:", pca.explained_variance_ratio_)