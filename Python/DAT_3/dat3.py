#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""

# Question 1 below
print("\n=== Logistic Regression Analysis ===\n")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.compose import ColumnTransformer
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report, accuracy_score, precision_recall_curve
from sklearn.inspection import permutation_importance
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from copy import deepcopy

file_path = "diabetes.csv"
df = pd.read_csv(file_path)
print("Original shape:", df.shape)

# Drop zodiac (irrelevant)
df = df.drop(columns=["Zodiac"])
print("After dropping zodiac:", df.shape)

# Coerce all to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Drop rows missing diabetes outcome


# ===========================
# Feature groups
# ===========================

target = "Diabetes"

# Already-binary variables → keep as-is
binary_vars = [
    'HighBP', 'HighChol', 'Smoker', 'Stroke', 'Myocardial',
    'PhysActivity', 'Fruit', 'Vegetables', 'HeavyDrinker',
    'HasHealthcare', 'NotAbleToAffordDoctor', 'HardToClimbStairs'
]

# Numerical variables
numeric_vars = ['BMI', 'MentalHealth', 'PhysicalHealth']

# Ordinal/categorical → one-hot encode
categorical_vars = [
    'BiologicalSex',                # encoded 1/2, but still categorical
    'GeneralHealth',         # categorical (1–5)
    'AgeBracket',
    'EducationBracket',
    'IncomeBracket'
]

# Define X, y
X = df[binary_vars + numeric_vars + categorical_vars]
y = df[target].astype(int)

# ===========================
# Preprocessing: 
#   - Numeric: impute + scale
#   - Categorical: impute + one-hot encode
# ===========================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(drop="first", handle_unknown="ignore"))
])

preprocess = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_vars),
        ("cat", categorical_transformer, categorical_vars),
        ("bin", "passthrough", binary_vars)
    ]
)

# ===========================
# Full modeling pipeline
# ===========================

logreg_pipeline = Pipeline(steps=[
    ("preprocess", preprocess),
    ("logreg", LogisticRegression(max_iter=3000))
])

# ===========================
# Train/test split
# ===========================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

# Fit model
logreg_pipeline.fit(X_train, y_train)

# Predict
y_pred_prob = logreg_pipeline.predict_proba(X_test)[:, 1]
y_pred = (y_pred_prob >= 0.5).astype(int)

# ===========================
# Performance metrics
# ===========================

auc = roc_auc_score(y_test, y_pred_prob)
acc = accuracy_score(y_test, y_pred)

print(f"\nAUC: {auc:.4f} | Accuracy: {acc:.4f}")
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred, digits=3))

# ===========================
# Permutation importance
# ===========================

# ---------------- Robust extraction of post-transform feature names ----------------

# ---------------- Simple, robust construction of transformed feature names ----------------

# ---------- Robust minimal replacement for the permutation+feature-name step ----------

preproc = logreg_pipeline.named_steps['preprocess']
logreg = logreg_pipeline.named_steps['logreg']

# 1) Transform X_test using the fitted preprocessor (exact same transform used in pipeline)
X_test_trans = preproc.transform(X_test)

# get number of transformed columns
if sparse.issparse(X_test_trans):
    n_trans_cols = X_test_trans.shape[1]
else:
    n_trans_cols = np.asarray(X_test_trans).shape[1]

print("Transformed X_test shape:", X_test_trans.shape)
print("Number of transformed columns:", n_trans_cols)

# 2) Run permutation importance on the trained logistic estimator using the transformed data
# (doing this on the estimator + transformed array guarantees the lengths align)
perm = permutation_importance(
    logreg,                # trained logistic regression (not the pipeline)
    X_test_trans,          # already-transformed feature matrix
    y_test,
    scoring='roc_auc',
    n_repeats=10,
    random_state=42,
    n_jobs=-1
)

print("Permutation importances computed. Length:", perm.importances_mean.shape[0])

# 3) Build feature names exactly matching the transformed columns (use the fitted OneHotEncoder)
constructed_names = []

# numeric vars (each remains 1 column)
constructed_names.extend(numeric_vars)

# categorical vars: use get_feature_names_out from the fitted OneHotEncoder inside the pipeline
ohe = preproc.named_transformers_['cat'].named_steps['onehot']
# get_feature_names_out needs the input feature names
try:
    cat_names = list(ohe.get_feature_names_out(categorical_vars))
except Exception:
    # fallback: use categories_ to build names (taking into account drop='first')
    cat_names = []
    for var_name, cats in zip(categorical_vars, ohe.categories_):
        cats_use = list(cats)
        # if drop='first', remove the first category
        if getattr(ohe, "drop", None) == 'first' or hasattr(ohe, "drop_idx_"):
            cats_use = cats_use[1:]
        for cat in cats_use:
            cat_names.append(f"{var_name}_{cat}")
constructed_names.extend(cat_names)

# passthrough binary vars
constructed_names.extend(binary_vars)

print("Constructed feature name count:", len(constructed_names))

# Diagnostic: print first 40 names and the last 10
print("Constructed names (first 20):", constructed_names[:20])
print("Constructed names (last 10):", constructed_names[-10:])

# 4) Sanity check: lengths must match
if len(constructed_names) != n_trans_cols:
    print("ERROR: constructed feature names length DOES NOT MATCH transformed columns.")
    print(f"len(constructed_names) = {len(constructed_names)}, n_trans_cols = {n_trans_cols}")
    # Print helpful diagnostics for debugging
    print("ohe.categories_ by categorical var:")
    for v, cats in zip(categorical_vars, ohe.categories_):
        print(v, "->", list(cats))
    raise ValueError("Feature name length mismatch. See diagnostics above.")

# 5) Build importance_df now that lengths are guaranteed to match
importance_df = pd.DataFrame({
    "feature": constructed_names,
    "mean_importance": perm.importances_mean,
    "std_importance": perm.importances_std
}).sort_values("mean_importance", ascending=False).reset_index(drop=True)

best_feature = importance_df.iloc[0]["feature"]
print(f"\nMost influential predictor (largest AUC drop if permuted): {best_feature}")

print("\nTop 10 permutation-important features:")
print(importance_df.head(10).to_string(index=False))


# ===========================
# Visual 1: ROC + PR curves
# ===========================

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# ROC
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
ax[0].plot(fpr, tpr, label=f"AUC={auc:.3f}")
ax[0].plot([0,1],[0,1],'k--')
ax[0].set_title("ROC Curve")
ax[0].set_xlabel("False Positive Rate")
ax[0].set_ylabel("True Positive Rate")
ax[0].grid(alpha=0.3)

# PR
prec, rec, _ = precision_recall_curve(y_test, y_pred_prob)
ax[1].plot(rec, prec, color="orange")
ax[1].set_title("Precision–Recall Curve")
ax[1].set_xlabel("Recall")
ax[1].set_ylabel("Precision")
ax[1].grid(alpha=0.3)

plt.suptitle("Model Performance — Logistic Regression (One-Hot Encoded Predictors)", fontsize=14)
plt.tight_layout()
plt.show()

# ===========================
# Visual 2: Permutation Importance
# ===========================

plt.figure(figsize=(10,6))
sns.barplot(
    data=importance_df.head(10),
    x="mean_importance",
    y="feature",
    palette="coolwarm"
)
plt.title("Top 10 Most Influential Predictors (Permutation Drop in AUC)")
plt.xlabel("Mean Drop in AUC When Shuffled")
plt.ylabel("Feature")
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

# ===========================
# Coefficient summary (standardized)
# ===========================

coef = logreg_pipeline.named_steps["logreg"].coef_[0]
coef_series = pd.Series(coef, index=constructed_names).sort_values(key=np.abs, ascending=False)

print("\nTop standardized logistic regression coefficients:")
print(coef_series.head(10))

# Question 2 below
##########################################################################################

print("\n=== SVM Analysis ===\n")

# 1) Build SVM pipeline reusing preprocess (so one-hot encoding is shared)
base_svc = LinearSVC(max_iter=10000, dual=False, random_state=42)
calibrated_svc = CalibratedClassifierCV(estimator=base_svc, method='sigmoid', cv=5)
svm_pipeline = Pipeline([
    ('preproc', preprocess),
    ('clf', calibrated_svc)
])

# Train/test split reuses X, y defined earlier but we'll re-split for SVM section (you already have X_train, X_test)
# Assuming X_train, X_test, y_train, y_test exist (from logistic section). If not, redo the split:
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

svm_pipeline.fit(X_train, y_train)

# Evaluate
y_prob = svm_pipeline.predict_proba(X_test)[:, 1]
y_pred = svm_pipeline.predict(X_test)
auc = roc_auc_score(y_test, y_prob)
print(f"\nLinearSVC (calibrated) AUC: {auc:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification report:\n", classification_report(y_test, y_pred, digits=3))

preproc_fitted = svm_pipeline.named_steps['preproc']

# --- (A) Permutation importances correspond to original raw features when run on the pipeline ---
perm = permutation_importance(svm_pipeline, X_test, y_test, scoring='roc_auc', n_repeats=20, random_state=42, n_jobs=-1)

# Build importance_df directly from original feature names (X_test.columns) — lengths will match
orig_feature_names = list(X_test.columns)
if perm.importances_mean.shape[0] != len(orig_feature_names):
    print("WARNING: perm length", perm.importances_mean.shape[0], "does not equal original feature count", len(orig_feature_names))
importance_df = pd.DataFrame({
    'feature': orig_feature_names,
    'mean_drop_auc': perm.importances_mean,
    'std': perm.importances_std
}).sort_values('mean_drop_auc', ascending=False).reset_index(drop=True)

print("\nTop features by permutation drop in AUC (original features):")
print(importance_df.head(15).to_string(index=False))

best_predictor = importance_df.loc[0, 'feature']
print("\nMost influential predictor (permutation importance):", best_predictor)

# --- (B) Build 'parents' list (one parent per transformed column) correctly and robustly ---
# We need the exact number of transformed columns and the parent for each column, in order.
X_test_trans = preproc_fitted.transform(X_test)
if sparse.issparse(X_test_trans):
    n_trans_cols = X_test_trans.shape[1]
else:
    n_trans_cols = np.asarray(X_test_trans).shape[1]
print("DEBUG: Transformed X_test shape:", X_test_trans.shape, "=>", n_trans_cols, "columns")

parents = []  # will hold exactly n_trans_cols entries
# Iterate through the fitted transformers_ in order
for name, transformer, cols in preproc_fitted.transformers_:
    # cols is either a list of column names (for transformers we passed) or a slice
    if transformer == 'drop':
        continue
    if transformer == 'passthrough':
        # passthrough columns: each produces one column with same name
        parents.extend(list(cols))
        continue

    # If transformer is a Pipeline we want the last step (the estimator that does the actual transform)
    t = transformer
    if isinstance(transformer, Pipeline):
        t = transformer.steps[-1][1]

    # Numeric transformer (imputer+scaler) produces exactly one column per numeric input
    if name == 'num':
        parents.extend(list(cols))
        continue

    # Categorical transformer (imputer+onehot): use the fitted OneHotEncoder to see categories per original var
    if name == 'cat':
        # t should be OneHotEncoder
        ohe = t
        # categories_ is a list aligned with the input categorical columns "cols"
        for var_name, cats in zip(cols, ohe.categories_):
            # number of dummy columns produced for this var depends on drop setting
            if getattr(ohe, "drop", None) is not None or hasattr(ohe, "drop_idx_"):
                n_dummies = max(0, len(cats) - 1)  # drop='first'
            else:
                n_dummies = len(cats)
            parents.extend([var_name] * n_dummies)
        continue

    # Passthrough / other fallback: assume one output per input col
    parents.extend(list(cols))

# Sanity check
print("DEBUG: built parents length:", len(parents))
if len(parents) != n_trans_cols:
    print("ERROR: parents length mismatch:", len(parents), "vs transformed cols", n_trans_cols)
    # print some diagnostics for debugging
    print("preproc_fitted.transformers_:", [(t[0], type(t[1]), t[2]) for t in preproc_fitted.transformers_])
    print("ohe.categories_ for cat transformer:", preproc_fitted.named_transformers_['cat'].named_steps['onehot'].categories_)
    raise ValueError("Parents length mismatch. Aborting. See diagnostics above.")

# --- (C) Fit a standalone LinearSVC on transformed training data and get coefficients ---
X_train_trans = preproc_fitted.transform(X_train)
if sparse.issparse(X_train_trans):
    X_train_trans = X_train_trans.toarray()

svc_for_coef = LinearSVC(max_iter=10000, dual=False, random_state=42)
svc_for_coef.fit(X_train_trans, y_train)
coefs = svc_for_coef.coef_.ravel()
print("DEBUG: coef length:", len(coefs))

if len(coefs) != n_trans_cols:
    raise ValueError(f"Length mismatch: coef length {len(coefs)} != n_trans_cols {n_trans_cols}")

# Build coef_df using the reliable parents list
transformed_names = [f"t{i}" for i in range(n_trans_cols)]
coef_df = pd.DataFrame({
    'trans_feature': transformed_names,
    'coef': coefs,
    'parent': parents
})

# Aggregate coefficients by parent (sum to keep sign)
agg_coef = coef_df.groupby('parent', as_index=False).agg(
    coef_sum=('coef', 'sum'),
    coef_absmean=('coef', lambda s: np.mean(np.abs(s))),
    coef_count=('coef', 'count')
).sort_values('coef_sum', key=lambda s: s.abs(), ascending=False).reset_index(drop=True)

print("\nTop aggregated coefficients (by summed signed effect):")
print(agg_coef.head(15).to_string(index=False))

# --- (D) Radar chart from aggregated coefficients ---
K = min(10, agg_coef.shape[0])
topk = agg_coef.head(K)
labels = topk['parent'].tolist()
values = topk['coef_sum'].tolist()

angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
angles_wrap = angles + angles[:1]
values_wrap = values + values[:1]

plt.figure(figsize=(7,7))
ax = plt.subplot(111, polar=True)
ax.plot(angles_wrap, values_wrap, marker='o')
ax.fill(angles_wrap, values_wrap, alpha=0.25)
ax.set_xticks(angles)
ax.set_xticklabels(labels, fontsize=10)
ax.set_title("Top aggregated LinearSVC weights by feature", pad=20)
plt.show()

# --- (E) Partial dependence for top 2 aggregated parents (as before) ---
def pdp_for_feature(pipeline, X_base, feature_name, grid=None):
    X_copy = X_base.copy().reset_index(drop=True)
    if feature_name in numeric_vars:
        if grid is None:
            lo, hi = np.percentile(X_copy[feature_name].dropna(), [1, 99])
            xs = np.linspace(lo, hi, 50)
        else:
            xs = np.array(grid)
        ys = []
        for x in xs:
            Xc = X_copy.copy()
            Xc[feature_name] = x
            probs = pipeline.predict_proba(Xc)[:, 1]
            ys.append(probs.mean())
        return xs, np.array(ys)
    if feature_name in categorical_vars:
        ohe = preproc_fitted.named_transformers_['cat'].named_steps['onehot']
        cats = list(ohe.categories_[categorical_vars.index(feature_name)])
        xs = cats
        ys = []
        for cat in xs:
            Xc = X_copy.copy()
            Xc[feature_name] = cat
            probs = pipeline.predict_proba(Xc)[:, 1]
            ys.append(probs.mean())
        return xs, np.array(ys)
    if feature_name in binary_vars:
        xs = [0,1]
        ys = []
        for x in xs:
            Xc = X_copy.copy()
            Xc[feature_name] = x
            probs = pipeline.predict_proba(Xc)[:, 1]
            ys.append(probs.mean())
        return xs, np.array(ys)
    raise ValueError("Unknown feature for PDP")

top2_feats = agg_coef['parent'].iloc[:2].tolist()
print("Top 2 aggregated features for PDP (by aggregated coef):", top2_feats)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))
for ax, feat in zip(axes, top2_feats):
    xs, ys = pdp_for_feature(svm_pipeline, X_test, feat)
    if feat in numeric_vars:
        ax.plot(xs, ys, marker='o')
        ax.set_xlabel(feat)
    else:
        ax.bar([str(x) for x in xs], ys)
        ax.set_xlabel(feat)
    ax.set_ylabel('Predicted P(diabetes)')
    ax.set_title(f'Partial dependence: {feat}')
plt.suptitle('Partial Dependence of SVM on Two Strongest Predictors', fontsize=14)
plt.tight_layout()
plt.show()

#Question 3

##########################################################################################

print("\n=== Indiv Decision Tree Analysis ===\n")

# 1) Build pipeline reusing the same preprocess (one-hot + scaling + passthrough)
tree_clf = DecisionTreeClassifier(max_depth=10, random_state=42)   # shallow for interpretability
tree_pipeline = Pipeline([
    ('preproc', preprocess),   # same ColumnTransformer used earlier
    ('clf', tree_clf)
])

# Fit tree pipeline on training data (reuse X_train, y_train)
tree_pipeline.fit(X_train, y_train)

# Predict & evaluate
y_prob_tree = tree_pipeline.predict_proba(X_test)[:, 1]
y_pred_tree = tree_pipeline.predict(X_test)

auc_tree = roc_auc_score(y_test, y_prob_tree)
print("\n=== Single Decision Tree ===")
print(f"AUC: {auc_tree:.4f}")
print("Confusion matrix:\n", confusion_matrix(y_test, y_pred_tree))
print("\nClassification report:\n", classification_report(y_test, y_pred_tree, digits=3))

# 2) Permutation importance (original-feature level) on the pipeline
perm_tree = permutation_importance(tree_pipeline, X_test, y_test, scoring='roc_auc', n_repeats=20, random_state=42, n_jobs=-1)
orig_names = list(X_test.columns)
importance_df_tree = pd.DataFrame({
    'feature': orig_names,
    'mean_drop_auc': perm_tree.importances_mean,
    'std': perm_tree.importances_std
}).sort_values('mean_drop_auc', ascending=False).reset_index(drop=True)

print("\nTop features by permutation drop in AUC (original features):")
print(importance_df_tree.head(10).to_string(index=False))

best_by_perm = importance_df_tree.loc[0, 'feature']
print("\nBest predictor by permutation (original features):", best_by_perm)

# 3) Identify root split feature from the fitted tree (maps to a transformed column -> parent original feature)
preproc_fitted = tree_pipeline.named_steps['preproc']
clf_fitted = tree_pipeline.named_steps['clf']  # DecisionTreeClassifier

# transform X_test to see transformed dimensions
X_test_trans = preproc_fitted.transform(X_test)
if sparse.issparse(X_test_trans):
    n_trans_cols = X_test_trans.shape[1]
else:
    n_trans_cols = np.asarray(X_test_trans).shape[1]

# build parents list (one parent per transformed column) — robust method
parents = []
for name, transformer, cols in preproc_fitted.transformers_:
    if transformer == 'drop':
        continue
    if transformer == 'passthrough':
        parents.extend(list(cols))
        continue
    # handle Pipeline
    t = transformer
    if isinstance(transformer, Pipeline):
        t = transformer.steps[-1][1]
    if name == 'num':
        parents.extend(list(cols))
    elif name == 'cat':
        ohe = t
        for var_name, cats in zip(cols, ohe.categories_):
            # account for drop='first'
            if getattr(ohe, "drop", None) is not None or hasattr(ohe, "drop_idx_"):
                n_dummies = max(0, len(cats) - 1)
            else:
                n_dummies = len(cats)
            parents.extend([var_name] * n_dummies)
    else:
        parents.extend(list(cols))

# sanity check
if len(parents) != n_trans_cols:
    print("WARNING: parent mapping length mismatch:", len(parents), "vs transformed cols", n_trans_cols)
    # proceed but be careful

# The tree's root split uses feature index:
tree_obj = clf_fitted.tree_
root_feature_idx = tree_obj.feature[0]
if root_feature_idx == -2:
    root_parent = None
    print("Root is a leaf (no split).")
else:
    # parent name for that transformed column index
    root_parent = parents[root_feature_idx] if root_feature_idx < len(parents) else f"t{root_feature_idx}"
    root_threshold = tree_obj.threshold[0]

    # Convert threshold back to original units if the root_parent is numeric (in numeric_vars)
    threshold_readable = root_threshold
    if root_parent in numeric_vars:
        # scaler for numeric transformer
        scaler_num = preproc_fitted.named_transformers_['num'].named_steps['scaler']
        # numeric feature ordering corresponds to numeric_vars
        idx_numeric = numeric_vars.index(root_parent)
        mean_ = scaler_num.mean_[idx_numeric]
        scale_ = scaler_num.scale_[idx_numeric]
        threshold_readable = root_threshold * scale_ + mean_
        threshold_note = f"{threshold_readable:.3f} (original units)"
    else:
        # if categorical one-hot, threshold relates to dummy values (0/1) and split threshold ~0.5
        threshold_note = f"{root_threshold:.3f} (on dummy / transformed scale)"

    print(f"\nRoot split feature (transformed idx {root_feature_idx}) -> parent: {root_parent}")
    print(f"Root threshold (transformed scale): {root_threshold:.4f}; approx original: {threshold_note}")

# 4) Print concise best predictor result (choose root_parent as "best predictor by tree split")
print("\nDecision tree 'best predictor' (root split):", root_parent)
print("Best predictor by permutation importance (pipeline):", best_by_perm)

# 5) Visual: Plot tree (compact, colored by predicted class proba)
plt.figure(figsize=(18,8))
# Use parents as feature_names; if length mismatch, use generic t0..tN
feature_names_for_plot = parents if len(parents) == n_trans_cols else [f"t{i}" for i in range(n_trans_cols)]
plot_tree(clf_fitted, feature_names=feature_names_for_plot, class_names=['no','yes'], filled=True, rounded=True, fontsize=10)
plt.title("Decision Tree (max_depth=3) — nodes colored by majority class and gini")
plt.show()

# 6) Unique visual: Heatmap of predicted probability vs GeneralHealth (categories) and binned BMI
#    This shows where the tree assigns high risk in the joint space of a top categorical and numeric predictor.
heat_feat_cat = best_by_perm if best_by_perm in categorical_vars else 'GeneralHealth'  # prefer GeneralHealth if not present
heat_feat_num = 'BMI'

# build grid: categories by BMI bins
df_test = X_test.copy().reset_index(drop=True)
df_test['pred_prob'] = y_prob_tree  # predicted prob from tree pipeline (aligned by index)

# create BMI bins (10 bins within 1st-99th percentiles to avoid extremes)
lo, hi = np.percentile(df_test['BMI'].dropna(), [1, 99])
bins = np.linspace(lo, hi, 10)
df_test['BMI_bin'] = pd.cut(df_test['BMI'], bins=bins, include_lowest=True)

# grouped mean predicted prob
grp = df_test.groupby([heat_feat_cat, 'BMI_bin'])['pred_prob'].mean().unstack(level='BMI_bin')

plt.figure(figsize=(12,6))
sns.heatmap(grp, cmap='rocket_r', cbar_kws={'label':'Predicted P(diabetes)'}, linewidths=.5)
plt.xlabel('BMI bin')
plt.ylabel(heat_feat_cat)
plt.title(f'Heatmap: Tree predicted P(diabetes) by {heat_feat_cat} vs BMI (binned)')
plt.tight_layout()
plt.show()

# 7) Print final short summary
print("\n=== Summary ===")
print(f"AUC (Decision Tree): {auc_tree:.4f}")
print(f"Best predictor by tree root split: {root_parent}")
print(f"Best predictor by permutation importance (original features): {best_by_perm}")
print("Note: root split shows which single transformed column the tree used first; permutation importance measures overall effect on AUC.")

# Question 4

##########################################################################################

print("\n=== Random Forest Feature Ablation ===\n")

rf_pipeline = Pipeline([
    ('preproc', deepcopy(preprocess)),
    ('clf', RandomForestClassifier(n_estimators=500, random_state=42, n_jobs=-1))
])
rf_pipeline.fit(X_train, y_train)

y_prob_all = rf_pipeline.predict_proba(X_test)[:,1]
auc_all = roc_auc_score(y_test, y_prob_all)
print(f"Baseline Random Forest AUC (all features): {auc_all:.4f}")

feature_auc_drop = []

for feat in X_train.columns:
    print(f"Testing with neutral {feat}")
    X_train_mod = X_train.copy()
    X_test_mod = X_test.copy()
    
    # Neutralize feature
    if feat in numeric_vars:
        median_val = X_train_mod[feat].median()
        X_train_mod[feat] = median_val
        X_test_mod[feat] = median_val
    else:  # categorical/binary
        mode_val = X_train_mod[feat].mode()[0]
        X_train_mod[feat] = mode_val
        X_test_mod[feat] = mode_val
    
    # Predict with same pipeline
    y_prob_mod = rf_pipeline.predict_proba(X_test_mod)[:,1]
    auc_mod = roc_auc_score(y_test, y_prob_mod)
    
    feature_auc_drop.append((feat, auc_all - auc_mod))

feature_auc_drop_df = pd.DataFrame(feature_auc_drop, columns=['feature','auc_drop']).sort_values('auc_drop', ascending=False).reset_index(drop=True)
best_feature = feature_auc_drop_df.loc[0,'feature']

print("\nTop 10 features by AUC drop when neutralized:")
print(feature_auc_drop_df.head(10).to_string(index=False))
print(f"\nStrongest predictor: {best_feature}")
print(f"Baseline AUC: {auc_all:.4f}")


# --- 4) Creative visual: Heatmap of predicted probabilities for top 2 predictors ---
top2_feats = feature_auc_drop_df['feature'].iloc[:2].tolist()
feat_cat = top2_feats[0] if top2_feats[0] in categorical_vars else 'GeneralHealth'
feat_num = top2_feats[1] if top2_feats[1] in numeric_vars else 'BMI'

df_test = X_test.copy().reset_index(drop=True)
# Use rf_pipeline (fitted Random Forest pipeline) instead of undefined rf_all
df_test['pred_prob'] = rf_pipeline.predict_proba(df_test)[:,1]

# Bin numeric feature
lo, hi = np.percentile(df_test[feat_num].dropna(), [1, 99])
bins = np.linspace(lo, hi, 15)
df_test[f'{feat_num}_bin'] = pd.cut(df_test[feat_num], bins=bins, include_lowest=True)

# Group and pivot for heatmap
grp = df_test.groupby([feat_cat, f'{feat_num}_bin'])['pred_prob'].mean().unstack(level=f'{feat_num}_bin')

plt.figure(figsize=(12,6))
sns.heatmap(grp, cmap='rocket_r', linewidths=0.5, cbar_kws={'label':'Predicted P(diabetes)'})
plt.xlabel(f'{feat_num} bin')
plt.ylabel(feat_cat)
plt.title(f'Random Forest predicted P(diabetes) by {feat_cat} vs {feat_num} (feature ablation)')
plt.tight_layout()
plt.show()

# Extra Credit Part b

print("\n=== Extra Credit: Unobvious Lifestyle-Age Interaction Analysis ===\n")

# Copy X_test and get predicted probabilities from the fitted Random Forest
df_test = X_test.copy().reset_index(drop=True)
df_test['pred_prob'] = rf_pipeline.predict_proba(df_test)[:, 1]

# Create a combined lifestyle group feature for PhysActivity, Fruit, Vegetables
df_test['LifestyleGroup'] = df_test['PhysActivity'].astype(str) + "_" + \
                            df_test['Fruit'].astype(str) + "_" + \
                            df_test['Vegetables'].astype(str)

# Map AgeBracket to labels for readability
age_labels = {
    1:'18-24',2:'25-29',3:'30-34',4:'35-39',5:'40-44',6:'45-49',
    7:'50-54',8:'55-59',9:'60-64',10:'65-69',11:'70-74',12:'75-79',13:'80+'
}
df_test['AgeLabel'] = df_test['AgeBracket'].map(age_labels)

# --- 1) Grouped heatmap: Mean predicted diabetes probability by AgeBracket and LifestyleGroup ---
heatmap_data = df_test.groupby(['AgeLabel','LifestyleGroup'])['pred_prob'].mean().unstack()

plt.figure(figsize=(14,6))
sns.heatmap(heatmap_data, cmap='rocket_r', linewidths=0.5, cbar_kws={'label':'Predicted P(diabetes)'})
plt.xlabel('Lifestyle (PhysActivity_Fruit_Vegetables)')
plt.ylabel('Age Bracket')
plt.title('Predicted Diabetes Probability by Age and Lifestyle Group')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# --- 2) Interaction bar plot: highlight extremes in risk ---
# Find top 5 highest risk and top 5 lowest risk lifestyle-age combinations
grp = df_test.groupby(['AgeLabel','LifestyleGroup'])['pred_prob'].mean().reset_index()
top_risk = grp.nlargest(5, 'pred_prob')
low_risk = grp.nsmallest(5, 'pred_prob')
highlight_df = pd.concat([top_risk, low_risk])

plt.figure(figsize=(10,6))
sns.barplot(data=highlight_df, x='AgeLabel', y='pred_prob', hue='LifestyleGroup', dodge=True, palette='coolwarm')
plt.ylabel('Predicted P(diabetes)')
plt.xlabel('Age Bracket')
plt.title('Highest and Lowest Predicted Diabetes Risk by Age & Lifestyle')
plt.legend(title='Lifestyle Group', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

print("Observation: Middle-aged individuals (AgeBracket 6-10) with no physical activity and poor fruit/vegetable " +
      "consumption have disproportionately high predicted risk, whereas combinations with either physical activity " +
      "or fruit/vegetable intake show clear protective effects. This interaction is not captured by main-effect " +
      "importance metrics alone.")








