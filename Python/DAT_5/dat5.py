#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 6 20:38:47 2025

@author: fran-pellegrino
"""

print("Deliverable 1: PCA\n\n")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.manifold import MDS
from sklearn.metrics import pairwise_distances
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

RANDOM_SEED = 13639406
np.random.seed(RANDOM_SEED)

CSV_PATH = "wines.csv"   # <-- put your file path here if needed

df = pd.read_csv(CSV_PATH)

print("Shape of dataset:", df.shape)

X = df.values  # shape (178, 13)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# -----------------------------
# 3. PCA
# -----------------------------
pca = PCA(random_state=RANDOM_SEED)
X_pca = pca.fit_transform(X_scaled)

explained_variance_ratio = pca.explained_variance_ratio_
eigenvalues = pca.explained_variance_

# -----------------------------
# 4. NUMBER OF EIGENVALUES > 1
# -----------------------------
num_eigen_gt1 = np.sum(eigenvalues > 1)

print("\nEigenvalues:\n", eigenvalues)
print("\nExplained variance ratios:\n", explained_variance_ratio)

print("\n✅ Number of eigenvalues > 1:", num_eigen_gt1)

# -----------------------------
# 5. VARIANCE EXPLAINED BY PC1 & PC2
# -----------------------------
var_pc1 = explained_variance_ratio[0]
var_pc2 = explained_variance_ratio[1]
total_var_2d = var_pc1 + var_pc2

print("\n✅ Variance explained by PC1:", round(var_pc1 * 100, 2), "%")
print("✅ Variance explained by PC2:", round(var_pc2 * 100, 2), "%")
print("✅ Total variance explained by first two PCs:",
      round(total_var_2d * 100, 2), "%")

# -----------------------------
# 6. 2D PCA PROJECTION PLOT
# -----------------------------
plt.figure(figsize=(8, 6))
plt.scatter(X_pca[:, 0], X_pca[:, 1], s=50)
plt.xlabel(f"PC1 ({var_pc1*100:.2f}% variance)")
plt.ylabel(f"PC2 ({var_pc2*100:.2f}% variance)")
plt.title("2D PCA Projection of Wine Dataset")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# 7. CREATIVE & INFORMATIVE EXTRA FIGURE:
# PCA LOADING HEATMAP (FEATURE → PC1/PC2 STRUCTURE)
# -----------------------------
loadings = pca.components_[:2]   # PC1 and PC2 loadings only
feature_names = df.columns

plt.figure(figsize=(10, 4))
plt.imshow(loadings, aspect='auto')
plt.yticks([0, 1], ["PC1", "PC2"])
plt.xticks(range(len(feature_names)), feature_names, rotation=45, ha="right")
plt.colorbar(label="Loading Strength")
plt.title("How Each Chemical Feature Contributes to PC1 and PC2")
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------------------
print("\n\nDeliverable 2: t-SNE\n\n")

RANDOM_SEED = 13639406

# If X_scaled (standardized features) not available from earlier part of the script, load and scale here:
try:
    Xs = X_scaled
    print("Using X_scaled from the PCA section.")
except NameError:
    print("X_scaled not found - reloading & standardizing wines.csv")
    df = pd.read_csv("wines.csv")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df.values)

n_samples = Xs.shape[0]
print(f"n_samples = {n_samples}")

# Define perplexities to explore (5 -> 150). Pick a diverse spread.
perplexities = [5, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, 125, 150]
perp_vals = []
kl_vals = []

# Run t-SNE for each perplexity and record final KL divergence
for perp in perplexities:
    # sanity check: perplexity must be < n_samples. Skip if too large (rare for 178 samples but safe).
    if perp >= n_samples:
        print(f"Skipping perplexity={perp} since it's >= n_samples ({n_samples})")
        continue

    print(f"Running t-SNE with perplexity={perp} ...")
    tsne = TSNE(
        n_components=2,
        perplexity=perp,
        random_state=RANDOM_SEED,
        init="pca",
        max_iter=1000,
        learning_rate="auto",
        metric="euclidean",
        verbose=0
    )
    X_tsne_tmp = tsne.fit_transform(Xs)

    # sklearn's TSNE exposes final KL divergence in attribute kl_divergence_
    try:
        kl = float(tsne.kl_divergence_)
    except Exception:
        # fallback: if the attribute name differs by version, try accessing private attributes (rare)
        kl = np.nan
        print("Warning: couldn't read 'kl_divergence_' from TSNE object; storing NaN for this run.")

    perp_vals.append(perp)
    kl_vals.append(kl)
    print(f"  -> final KL divergence: {kl:.6f}")

# Convert to numpy arrays
perp_vals = np.array(perp_vals)
kl_vals = np.array(kl_vals)

# Plot: Perplexity vs KL divergence
plt.figure(figsize=(8,5))
plt.plot(perp_vals, kl_vals, marker='o',linewidth=2)
plt.xlabel("t-SNE Perplexity")
plt.ylabel("Final KL divergence (fit objective)")
plt.title("Perplexity vs Final t-SNE KL-divergence")
plt.grid(True)
plt.tight_layout()
plt.show()


# -----------------------------
# t-SNE plot for perplexity = 20
# -----------------------------
perp20 = 20
if perp20 >= n_samples:
    raise ValueError(f"Selected perplexity {perp20} is >= n_samples ({n_samples}); choose smaller perplexity.")

print(f"\nGenerating final t-SNE 2D plot with perplexity={perp20} ...")
tsne20 = TSNE(
    n_components=2,
    perplexity=perp20,
    random_state=RANDOM_SEED,
    init="pca",
    max_iter=1000,
    learning_rate="auto",
    metric="euclidean",
    verbose=1
)
X_tsne_20 = tsne20.fit_transform(Xs)

# scatter plot
plt.figure(figsize=(8,6))
plt.scatter(X_tsne_20[:,0], X_tsne_20[:,1], s=50, alpha=0.9)
plt.title(f"t-SNE 2D projection (perplexity={perp20})")
plt.xlabel("t-SNE dim 1")
plt.ylabel("t-SNE dim 2")
plt.grid(False)
plt.tight_layout()
plt.show()

print("\n--- quick notes on expected KL vs Perplexity behavior ---")
print("Typically, as perplexity increases from very small values, the final KL divergence will")
print("tend to decrease (the model has easier time matching a slightly 'smoother' neighborhood distribution),")
print("up to a point where it may level off or even slightly increase as perplexity becomes so large")
print("that local structure is smoothed out and the objective changes shape. Inspect the saved plot to see")
print("exactly how KL depends on perplexity for this dataset.")


# --------------------------------------------------------------------------------------
print("\n\nDeliverable 3: \n\n")

RANDOM_SEED = 13639406

# Ensure standardized data exists
try:
    Xs = X_scaled
    print("Using X_scaled from earlier in the script.")
except NameError:
    print("X_scaled not available - loading & standardizing wines.csv")
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    df = pd.read_csv("wines.csv")
    scaler = StandardScaler()
    Xs = scaler.fit_transform(df.values)

n_samples = Xs.shape[0]

# 1) Compute pairwise distances in original space (high-d)
D_high = pairwise_distances(Xs, metric='euclidean')

# 2) Run MDS (metric) into 2D
mds = MDS(n_components=2,
          metric=True,
          n_init=4,
          max_iter=300,
          random_state=RANDOM_SEED,
          dissimilarity='precomputed',
          n_jobs=None)   # use precomputed distances below

# sklearn's MDS with dissimilarity='precomputed' expects a distance matrix for fit()
X_mds = mds.fit_transform(D_high)  # pass D_high as "dissimilarity" matrix
stress = mds.stress_

print(f"\n✅ MDS computed. Stress (raw) = {stress:.6f}")

# 3) 2D MDS scatter plot
plt.figure(figsize=(8,6))
plt.scatter(X_mds[:,0], X_mds[:,1], s=50, alpha=0.9)
plt.title(f"MDS 2D embedding (stress={stress:.4f})")
plt.xlabel("MDS dim 1")
plt.ylabel("MDS dim 2")
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Creative figure A: Shepard diagram (original distances vs embedded distances)
# -----------------------------
# compute pairwise distances in embedding
D_low = pairwise_distances(X_mds, metric='euclidean')

# flatten upper triangle (avoid duplicates and zeros)
iu = np.triu_indices(n_samples, k=1)
d_high_vals = D_high[iu]
d_low_vals = D_low[iu]

# Fit linear model for trend line and compute R^2
lm = LinearRegression().fit(d_high_vals.reshape(-1, 1), d_low_vals)
pred = lm.predict(d_high_vals.reshape(-1, 1))
r2 = r2_score(d_low_vals, pred)
slope = lm.coef_[0]
intercept = lm.intercept_

plt.figure(figsize=(7,6))
plt.scatter(d_high_vals, d_low_vals, s=8, alpha=0.4)
# plot identity line
lims = [0, max(d_high_vals.max(), d_low_vals.max())]
plt.plot(lims, lims, '--', linewidth=1.2, label='identity y=x')
# plot linear fit
xs = np.linspace(lims[0], lims[1], 200)
plt.plot(xs, slope*xs + intercept, '-', color='C1', linewidth=2, label=f'linear fit (R²={r2:.3f})')
plt.xlabel("High-dimensional Euclidean distance")
plt.ylabel("Low-dimensional (MDS) Euclidean distance")
plt.title("Shepard diagram: original vs MDS distances")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# -----------------------------
# Creative figure B: Per-point stress plotted on MDS scatter
# -----------------------------
# Per-point stress measure: for point i, s_i = sum_j (d_high_ij - d_low_ij)^2
per_point_sqerr = np.sum((D_high - D_low)**2, axis=1)
# normalize for plotting sizes/colors
norm_stress = (per_point_sqerr - per_point_sqerr.min()) / (per_point_sqerr.ptp() + 1e-12)

plt.figure(figsize=(8,6))
# use a colormap to emphasize high stress
sc = plt.scatter(X_mds[:,0], X_mds[:,1],
                 s=40 + 160 * norm_stress,          # base size plus scaled by stress
                 c=norm_stress, cmap='inferno', alpha=0.9)
plt.colorbar(sc, label='normalized per-point stress (higher = poorer fit)')
plt.title("MDS 2D embedding with per-point stress (size & color ~ stress)")
plt.xlabel("MDS dim 1")
plt.ylabel("MDS dim 2")
plt.grid(False)
plt.tight_layout()
plt.show()

print("\nMDS diagnostic summary:")
print(f"  - stress_: {stress:.6f}")
print(f"  - Shepard linear fit slope: {slope:.4f}, intercept: {intercept:.4f}, R^2: {r2:.4f}")
print(f"  - per-point stress: mean={per_point_sqerr.mean():.6f}, max={per_point_sqerr.max():.6f}")

# ---------------------------------------------------------------------------------------
print("\n\nDeliverable 4: Silhouette + kMeans on t-SNE\n\n")

np.random.seed(13639406)

k_range = range(2, 11)
sil_scores = []

print("\nSilhouette Analysis on t-SNE Output:")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=13639406, n_init=10)
    labels = kmeans.fit_predict(X_tsne_20)
    score = silhouette_score(X_tsne_20, labels)
    sil_scores.append(score)
    print(f"  k = {k} | Silhouette Score = {score:.5f}")

# Optimal k
optimal_k = k_range[np.argmax(sil_scores)]
print(f"\n✅ Optimal number of clusters (k) = {optimal_k}")

# PLOT SILHOUETTE SCORES

plt.figure()
plt.plot(list(k_range), sil_scores, marker='o')
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Method on t-SNE Embedding")
plt.grid(True)
plt.show()

# INAL KMEANS WITH OPTIMAL k

kmeans_final = KMeans(n_clusters=optimal_k, random_state=13639406, n_init=10)
final_labels = kmeans_final.fit_predict(X_tsne_20)
centers = kmeans_final.cluster_centers_

# TOTAL EUCLIDEAN DISTANCE TO CENTERS

total_distance = 0
for i in range(len(X_tsne_20)):
    center = centers[final_labels[i]]
    total_distance += np.linalg.norm(X_tsne_20[i] - center)

print(f"\n✅ Total sum of Euclidean distances to cluster centers: {total_distance:.6f}")


# CLUSTERED t-SNE PLOT

plt.figure()
plt.scatter(X_tsne_20[:, 0], X_tsne_20[:, 1], c=final_labels)
plt.scatter(centers[:, 0], centers[:, 1], marker='X', s=200)
plt.title(f"KMeans Clustering on t-SNE (k = {optimal_k})")
plt.xlabel("t-SNE Component 1")
plt.ylabel("t-SNE Component 2")
plt.grid(True)
plt.show()
