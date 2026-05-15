import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

#etf list/reading in data
tickers = [
    "ESGV","IVOO","IVOV","MGC","MGK","MGV","VB","VBK","VBR",
    "VDIG","VEA","VEU","VEXC","VFMF","VFMO","VFMV","VFQY","VFVA",
    "VGK","VIG","VIGI","VIOG","VIOO","VIOV","VO","VOE","VONE",
    "VONG","VONV","VOO","VOOG","VOT","VPL","VSGX","VSS","VT",
    "VTHR","VTI","VTV","VTWG","VTWO","VTWV","VUG","VUSG","VUSV",
    "VV","VWO","VXF","VXUS","VYM","VYMI"
]

#expanding on last analysis with similarity distances
holdings_rows = []

for etf in tickers:
    df = pd.read_csv(f"{etf}_holdings.csv")
    df = df[["TICKER", "% OF FUNDS*"]].copy()

    df["TICKER"] = df["TICKER"].astype(str).str.strip()
    df["% OF FUNDS*"] = (
        df["% OF FUNDS*"]
        .astype(str)
        .str.replace("%", "", regex=False)
        .str.replace("<", "", regex=False)
        .str.replace(",", "", regex=False)
        .str.replace("---", "0", regex=False)
    )
    df["% OF FUNDS*"] = pd.to_numeric(df["% OF FUNDS*"], errors="coerce").fillna(0)
    df["ETF"] = etf
    holdings_rows.append(df)

holdings_df = pd.concat(holdings_rows, ignore_index=True)

holdings_matrix = holdings_df.pivot_table(
    index="ETF",
    columns="TICKER",
    values="% OF FUNDS*",
    aggfunc="sum",
    fill_value=0
)

#normalize each ETF row to sum to 1
holdings_matrix = holdings_matrix.div(holdings_matrix.sum(axis=1), axis=0).fillna(0)

cosine_dist = pairwise_distances(holdings_matrix, metric="cosine")
euclidean_dist = pairwise_distances(holdings_matrix, metric="euclidean")
jaccard_dist = pairwise_distances((holdings_matrix > 0).to_numpy(), metric="jaccard")

distance_metrics = {
    "Cosine Distance": cosine_dist,
    "Euclidean Distance": euclidean_dist,
    "Jaccard Distance": jaccard_dist
}

for name, matrix in distance_metrics.items():
    dist_df = pd.DataFrame(matrix, index=holdings_matrix.index, columns=holdings_matrix.index)

    plt.figure(figsize=(14, 12))
    plt.imshow(dist_df)
    plt.colorbar()
    plt.xticks(range(len(dist_df.columns)), dist_df.columns, rotation=90)
    plt.yticks(range(len(dist_df.index)), dist_df.index)
    plt.title(f"{name} Between ETF Holdings")
    plt.tight_layout()
    plt.savefig(f"{name.lower().replace(' ', '_')}_heatmap.png", dpi=300, bbox_inches="tight")
    plt.show()

    pairs = []
    for i in range(len(dist_df.index)):
        for j in range(i + 1, len(dist_df.columns)):
            pairs.append({
                "ETF 1": dist_df.index[i],
                "ETF 2": dist_df.columns[j],
                "Distance": dist_df.iloc[i, j]
            })

    pairs_df = pd.DataFrame(pairs).sort_values("Distance").head(15)
    print(f"\nClosest ETF pairs by {name}:")
    print(pairs_df.round(4))

#method 1: kmeans clustering
sector_rows = []

for etf in tickers:
    df = pd.read_csv(f"{etf}_sectors.csv")
    df = df.iloc[:, [0, 1]].copy()
    df.columns = ["Sector", "Weight"]
    df["Weight"] = (df["Weight"].astype(str).str.replace("%", "", regex=False).astype(float))
    df["ETF"] = etf
    sector_rows.append(df)

sector_df = pd.concat(sector_rows, ignore_index=True)
sector_df = sector_df.groupby(["ETF", "Sector"], as_index=False)["Weight"].sum()

sector_matrix = sector_df.pivot_table(
    index="ETF",
    columns="Sector",
    values="Weight",
    fill_value=0
)

#scale for kmeans/pca
scaler = StandardScaler()
X_scaled = scaler.fit_transform(sector_matrix)

#elbow plot
inertia = []
for k in range(1, 11):
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertia.append(km.inertia_)

plt.figure(figsize=(7, 5))
plt.plot(range(1, 11), inertia, marker="o")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.title("Elbow Plot")
plt.tight_layout()
plt.savefig("elbow_plot.png", dpi=300)
plt.show()

#final clustering
kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
clusters = kmeans.fit_predict(X_scaled)

cluster_assignments = pd.DataFrame({
    "ETF": sector_matrix.index,
    "Cluster": clusters
})

print("\nETFs by Cluster:")
print(cluster_assignments.sort_values("Cluster").to_string(index=False))

#clean cluster table
cluster_groups = (cluster_assignments.sort_values("Cluster").groupby("Cluster")["ETF"].apply(lambda x: ", ".join(x)).reset_index())

print("\nCluster Membership Table:")
print(cluster_groups.to_string(index=False))
cluster_summary = sector_matrix.copy()
cluster_summary["Cluster"] = clusters
cluster_summary_table = cluster_summary.groupby("Cluster").mean().round(2)

print("\nAverage Sector Exposure by Cluster:")
print(cluster_summary_table)

#PCA scatter
pca = PCA(n_components=2)
pca_data = pca.fit_transform(X_scaled)

pca_df = pd.DataFrame(pca_data, columns=["PC1", "PC2"], index=sector_matrix.index)
pca_df["Cluster"] = clusters

plt.figure(figsize=(10, 7))
plt.scatter(pca_df["PC1"], pca_df["PC2"], c=pca_df["Cluster"], cmap="Set1", alpha=0.75)

for etf in pca_df.index:
    plt.text(pca_df.loc[etf, "PC1"], pca_df.loc[etf, "PC2"], etf, fontsize=8)

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.title("ETF Clusters Based on Sector Composition")
plt.tight_layout()
plt.savefig("pca_cluster_plot.png", dpi=300)
plt.show()

# medthod 2: supervised learning/linear regression
returns_rows = []

for etf in tickers:
    r = pd.read_csv(f"{etf}_returns.csv", skiprows=2, header=None)
    nav_row = r[r[1].astype(str).str.contains("NAV", regex=False)]

    if nav_row.empty:
        continue

    val = str(nav_row.iloc[0, 6]).strip()
    if val == "---":
        continue

    returns_rows.append({
        "ETF": etf,
        "Return": float(val.replace("%", ""))
    })

returns_df = pd.DataFrame(returns_rows)

model_data = sector_matrix.merge(returns_df, left_index=True, right_on="ETF")

X = model_data.drop(columns=["ETF", "Return"])
y = model_data["Return"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print("\nSupervised Learning: Linear Regression")
print("Problem type: Regression")
print("Features: Sector weights")
print("Target: 5-year NAV return\n")
print(f"MSE:  {mse:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R^2:  {r2:.4f}")

results = model_data.loc[X_test.index, ["ETF"]].copy()
results["Actual Return"] = y_test.values
results["Predicted Return"] = y_pred
results["Abs Error"] = (results["Actual Return"] - results["Predicted Return"]).abs()

print("\nTop 5 Largest Errors:")
print(results.sort_values("Abs Error", ascending=False).head(5).round(4).to_string(index=False))

# predicted vs actual
plt.figure(figsize=(9, 7))
plt.scatter(y_pred, y_test)

for i, etf in enumerate(results["ETF"]):
    plt.text(y_pred[i], y_test.iloc[i], etf, fontsize=8)

min_val = min(y.min(), y_pred.min())
max_val = max(y.max(), y_pred.max())

plt.plot([min_val, max_val], [min_val, max_val], linestyle="--")
plt.xlabel("Predicted 5-Year Return")
plt.ylabel("Actual 5-Year Return")
plt.title("Predicted vs Actual ETF Returns")
plt.tight_layout()
plt.savefig("predicted_actual_returns_scatter.png", dpi=300)
plt.show()

#absolute error bar chart
error_plot = results.sort_values("Abs Error", ascending=False)

plt.figure(figsize=(10, 6))
plt.bar(error_plot["ETF"], error_plot["Abs Error"])
plt.xticks(rotation=90)
plt.xlabel("ETF")
plt.ylabel("Absolute Error")
plt.title("Absolute Prediction Error by ETF (Test Set)")
plt.tight_layout()
plt.savefig("prediction_error_by_etf.png", dpi=300)
plt.show()

#connection between methods
#prediction by cluster 
results = results.merge(cluster_assignments, on="ETF", how="left")

cluster_pred_summary = (
    results.groupby("Cluster")[["Actual Return", "Predicted Return", "Abs Error"]].mean().round(3).sort_index())

print("\nCluster-Level Prediction Summary (Test Set):")
print(cluster_pred_summary)