import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import rand_score, homogeneity_score, completeness_score, silhouette_score
from sklearn.datasets import load_iris

from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Load the Iris dataset
iris = load_iris()
X = iris.data  # Feature data
T = iris.target  # True class labels


# Step 2: Train KMeans on Each Transformed Dataset
# We will train KMeans with 3 clusters (n_clusters=3) and random_state=0, then compute the clustering metrics for:

# PCA-transformed data (X_pca)
pca = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components = 2, random_state = 0) )
])
X_pca = pca.fit_transform(X)

# LDA-transformed data (X_lda)
lda = LinearDiscriminantAnalysis(n_components=2)
lda.fit(X, T)
X_lda = lda.transform(X)

# t-SNE-transformed data (X_tsne)
tsne = TSNE(n_components=2, random_state=0)
X_tsne = tsne.fit_transform(X)


def evaluate_clustering(X_transformed, T, method_name):
    """
    Train KMeans clustering and compute evaluation metrics.

    Args:
        X_transformed: Transformed dataset (PCA, LDA, or t-SNE)
        T: True labels
        method_name: Name of the transformation method
    
    Returns:
        Pandas DataFrame with clustering scores
    """
    # Train KMeans on transformed data
    kmeans = KMeans(n_clusters=3, random_state=0, n_init=10)
    kmeans.fit(X_transformed)
    
    # Get predicted labels
    y_pred = kmeans.labels_

    # Compute clustering metrics
    r_score = rand_score(T, y_pred)
    h_score = homogeneity_score(T, y_pred)
    c_score = completeness_score(T, y_pred)
    s_score = silhouette_score(X_transformed, y_pred)

    # Store results in DataFrame
    scores = [[method_name, r_score, h_score, c_score, s_score]]
    df_columns = ["Method", "Rand Score", "Homogeneity", "Completeness", "Silhouette Score"]
    res = pd.DataFrame(scores, columns=df_columns)
    
    return res

# Step 4: Compute Scores for PCA, LDA, and t-SNE
# Now, we apply the function to each transformed dataset.

# Code: Run Evaluations

# Evaluate KMeans on PCA-transformed data
res_pca = evaluate_clustering(X_pca, T, "PCA")

# Evaluate KMeans on LDA-transformed data
res_lda = evaluate_clustering(X_lda, T, "LDA")

# Evaluate KMeans on t-SNE-transformed data
res_tsne = evaluate_clustering(X_tsne, T, "t-SNE")

# Combine results into one DataFrame
results = pd.concat([res_pca, res_lda, res_tsne], ignore_index=True)

# Display results
print(results)
