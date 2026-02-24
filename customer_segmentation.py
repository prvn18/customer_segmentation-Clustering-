import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# LOAD DATA
df = pd.read_csv("customers.csv")

print("\nSample data:")
print(df.head())

# FEATURE TYPES
numeric_features = ["Age", "Income", "SpendingScore"]
categorical_features = ["Gender", "Region"]

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("cat", categorical_pipeline, categorical_features)
])

X_processed = preprocessor.fit_transform(df)

# SELECT CLUSTER COUNT
best_score = -1
best_k = 2

for k in range(2, 8):
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_processed)

    score = silhouette_score(X_processed, labels)

    if score > best_score:
        best_score = score
        best_k = k

print(f"\nChosen clusters: {best_k}")

# TRAIN MODEL-
kmeans = KMeans(n_clusters=best_k, random_state=42)
df["Cluster"] = kmeans.fit_predict(X_processed)

print("\nCluster profiles:")
print(df.groupby("Cluster")[numeric_features].mean())

# PCA VISUALIZATION
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)

plt.figure()
plt.scatter(X_pca[:, 0], X_pca[:, 1], c=df["Cluster"])
plt.title("Customer Segments")
plt.show()