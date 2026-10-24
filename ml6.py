
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Load dataset
df = pd.read_csv("iris.csv")
X = df.iloc[:, :-1].select_dtypes(include="number")

# Scale data
X = StandardScaler().fit_transform(X)

# K-Means clustering
model = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = model.fit_predict(X)

# Evaluate clustering
print("Silhouette Score:", silhouette_score(X, clusters))
print("Cluster Centers:\n", model.cluster_centers_)

# Plot clusters
plt.scatter(X[:, 0], X[:, 2], c=clusters)
plt.title("Iris K-Means Clustering")
plt.xlabel("Sepal Length")
plt.ylabel("Petal Length")
plt.show()
