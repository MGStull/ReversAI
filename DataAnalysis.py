import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
#Just manually split into test and train set for ease
csv_path = "C:\\Users\\chick\\Documents\\Code\\ReversAI\\Data\\othello_dataset_train.csv"
df = pd.read_csv(csv_path)
print(len(df))

#Data Details
    #First is game id ######
    #Then 
    #Combinations of an a = [a-h] and integer b=[1-8]  with format abababab as a string of moves alternating between white and black
numbers = '12345678'
letters = 'abcdefgh'
tokens = []
for a in letters:
    for b in numbers:
        tokens.append(a+b)

print(df.iloc[0:2,1],df.iloc[0:2,2])
print(tokens)


def chunk_string(s):
    return [s[i:i+2] for i in range(0,len(s),2)]

chunk_string(df.iloc[0:19999,2])
print(df.iloc[0,2])

#Next task:
# Do an PCA to get an idea for which moves correlate to winning

def moves_to_feature_vector(moves_str, max_moves=60):
    """
    Convert a sequence of moves into a feature vector.
    Each feature represents how many times that move was played.
    """
    moves = chunk_string(moves_str)
    
    # Create a vector where each position corresponds to a move token
    feature_vector = np.zeros(len(tokens))
    
    for move in moves[:max_moves]:  # Only consider first max_moves
        if move in tokens:
            idx = tokens.index(move)
            feature_vector[idx] += 1
    
    return feature_vector

# Convert all games to feature vectors
print("\nConverting games to feature vectors...")
feature_vectors = []
labels = []

for idx, row in df.iterrows():
    if idx % 5000 == 0:
        print(f"Processing row {idx}...")
    
    moves_str = row[2]  # Adjust column index if different
    label = row[1]      # Adjust column index for winner label (if exists)
    
    feature_vector = moves_to_feature_vector(moves_str)
    feature_vectors.append(feature_vector)
    labels.append(label)

X = np.array(feature_vectors)
y = np.array(labels)

print(f"\nFeature matrix shape: {X.shape}")
print(f"Labels shape: {y.shape}")

# Standardize the features (important for PCA)
print("\nStandardizing features...")
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
print("Applying PCA...")
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Print explained variance
cumsum_var = np.cumsum(pca.explained_variance_ratio_)
print(f"\nExplained variance by first 5 components: {pca.explained_variance_ratio_[:5]}")
print(f"Cumulative explained variance (first 10): {cumsum_var[:10]}")

# Find number of components for 95% variance
n_components_95 = np.argmax(cumsum_var >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components_95}")

# Plot explained variance
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
axes[0].plot(range(1, min(51, len(cumsum_var)+1)), 
             cumsum_var[:50], 'bo-', linewidth=2, markersize=4)
axes[0].axhline(y=0.95, color='r', linestyle='--', label='95% variance')
axes[0].set_xlabel('Number of Components')
axes[0].set_ylabel('Cumulative Explained Variance')
axes[0].set_title('PCA Scree Plot')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Individual variance by component
axes[1].plot(range(1, min(51, len(pca.explained_variance_ratio_)+1)), 
             pca.explained_variance_ratio_[:50], 'go-', linewidth=2, markersize=4)
axes[1].set_xlabel('Component')
axes[1].set_ylabel('Explained Variance Ratio')
axes[1].set_title('Variance Explained by Each Component')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('pca_variance.png', dpi=100)
plt.show()

# Analyze which moves are important in first 2 principal components
print("\n=== TOP CONTRIBUTING MOVES TO PRINCIPAL COMPONENTS ===")

for pc_idx in range(min(3, len(pca.components_))):
    print(f"\nPrincipal Component {pc_idx + 1}:")
    
    # Get the loadings (contributions of each move)
    loadings = pca.components_[pc_idx]
    
    # Find top positive and negative contributors
    top_positive_idx = np.argsort(loadings)[-5:][::-1]
    top_negative_idx = np.argsort(loadings)[:5]
    
    print("  Top positive contributors (favor winning):")
    for idx in top_positive_idx:
        print(f"    {tokens[idx]}: {loadings[idx]:.4f}")
    
    print("  Top negative contributors:")
    for idx in top_negative_idx:
        print(f"    {tokens[idx]}: {loadings[idx]:.4f}")

# Visualize first two components colored by outcome
if len(np.unique(y)) <= 3:  # If there are reasonable number of classes
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create color map for different outcomes
    unique_labels = np.unique(y)
    colors = ['red', 'blue', 'green'][:len(unique_labels)]
    
    for label, color in zip(unique_labels, colors):
        mask = y == label
        ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
                  c=color, label=str(label), alpha=0.5, s=10)
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
    ax.set_title('PCA: Games Colored by Outcome')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('pca_scatter.png', dpi=100)
    plt.show()

print("\nPCA analysis complete! Plots saved as 'pca_variance.png' and 'pca_scatter.png'")


from sklearn.metrics import silhouette_score

for k in K_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(X_pca)
    inertias.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_pca, kmeans.labels_))
    print(f"k={k}: Inertia={kmeans.inertia_:.2f}, Silhouette Score={silhouette_scores[-1]:.4f}")

# Plot elbow curve
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(K_range, inertias, 'bo-', linewidth=2, markersize=8)
axes[0].set_xlabel('Number of Clusters (k)')
axes[0].set_ylabel('Inertia')
axes[0].set_title('Elbow Method for Optimal k')
axes[0].grid(True, alpha=0.3)

axes[1].plot(K_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
axes[1].set_xlabel('Number of Clusters (k)')
axes[1].set_ylabel('Silhouette Score')
axes[1].set_title('Silhouette Score by Number of Clusters')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('elbow_silhouette.png', dpi=100)
plt.show()

# Use optimal k (let's try k=4 as a reasonable choice)
optimal_k = 4
print(f"\n=== K-Means Clustering with k={optimal_k} ===")
kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
cluster_labels = kmeans.fit_predict(X_pca)

# Analyze clusters
print(f"\nCluster sizes:")
for i in range(optimal_k):
    count = np.sum(cluster_labels == i)
    print(f"  Cluster {i}: {count} games ({100*count/len(cluster_labels):.1f}%)")

# Analyze win rates by cluster
print(f"\n=== Win Rates by Cluster ===")
unique_outcomes = np.unique(y)
print(f"Possible outcomes: {unique_outcomes}")

cluster_outcome_stats = {}
for cluster_id in range(optimal_k):
    mask = cluster_labels == cluster_id
    cluster_outcomes = y[mask]
    
    print(f"\nCluster {cluster_id}:")
    for outcome in unique_outcomes:
        outcome_count = np.sum(cluster_outcomes == outcome)
        outcome_pct = 100 * outcome_count / np.sum(mask)
        print(f"  {outcome}: {outcome_count} ({outcome_pct:.1f}%)")
    
    cluster_outcome_stats[cluster_id] = dict(zip(unique_outcomes, 
                                                  [np.sum(cluster_outcomes == o) for o in unique_outcomes]))

# Visualize clusters in 2D PCA space
fig, ax = plt.subplots(figsize=(12, 10))

# Plot by cluster
colors = ['red', 'blue', 'green', 'purple', 'orange', 'brown']
for cluster_id in range(optimal_k):
    mask = cluster_labels == cluster_id
    ax.scatter(X_pca[mask, 0], X_pca[mask, 1], 
              c=colors[cluster_id % len(colors)], 
              label=f'Cluster {cluster_id}', 
              alpha=0.6, s=20)

# Plot cluster centers
centers_2d = kmeans.cluster_centers_[:, :2]
ax.scatter(centers_2d[:, 0], centers_2d[:, 1], 
          c='black', marker='X', s=300, edgecolors='white', linewidths=2,
          label='Centroids')

ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
ax.set_title(f'K-Means Clustering (k={optimal_k}) - Games in PCA Space')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('kmeans_clusters.png', dpi=100)
plt.show()

# Analyze top moves in each cluster
print(f"\n=== Top Moves by Cluster ===")
for cluster_id in range(optimal_k):
    mask = cluster_labels == cluster_id
    cluster_vectors = X[mask]  # Original feature space
    mean_vector = np.mean(cluster_vectors, axis=0)
    
    # Find top moves
    top_move_idx = np.argsort(mean_vector)[-5:][::-1]
    
    print(f"\nCluster {cluster_id} - Top moves:")
    for idx in top_move_idx:
        print(f"  {tokens[idx]}: avg frequency {mean_vector[idx]:.2f}")

print("\n✓ Analysis complete! Check 'elbow_silhouette.png' and 'kmeans_clusters.png'")