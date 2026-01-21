#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

@author: fran-pellegrino
"""

import numpy as np
import pandas as pd
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


# labels
sports = ["Football", "Soccer", "Basketball", "Baseball", "Boxing", "Golf", "Chess"]

# Dissimilarity matrix (symmetric, diagonal = 0)
dissimilarity = np.array([
    [0, 2, 4, 6, 4, 7, 6],
    [2, 0, 3, 6, 5, 7, 5],
    [4, 3, 0, 6, 5, 5, 5],
    [6, 6, 6, 0, 6, 4, 7],
    [4, 5, 5, 6, 0, 7, 7],
    [7, 7, 5, 4, 7, 0, 7],
    [6, 5, 5, 7, 7, 7, 0]
])

df = pd.DataFrame(dissimilarity, index=sports, columns=sports)


# mds mapping
mds = MDS(
    n_components=2,
    dissimilarity="precomputed",
    random_state=42
)

coords = mds.fit_transform(dissimilarity)

# plotting
plt.figure(figsize=(8, 6))
plt.scatter(coords[:, 0], coords[:, 1])

for i, sport in enumerate(sports):
    plt.text(coords[i, 0] + 0.02, coords[i, 1] + 0.02, sport)

plt.xlabel("MDS Dimension 1")
plt.ylabel("MDS Dimension 2")
plt.title("2D MDS Map of Sports Dissimilarity")
plt.axhline(0, linewidth=0.5)
plt.axvline(0, linewidth=0.5)
plt.grid(True)

plt.show()


