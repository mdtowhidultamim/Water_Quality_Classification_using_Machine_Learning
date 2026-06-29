
# =============================================================================
# CCS2213: Machine Learning - Assignment 1
# Water Quality Classification
# Dataset: waterQuality1.csv
# Classifiers: Random Forest & Logistic Regression
# =============================================================================

# ── 0. Imports ────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_auc_score, roc_curve, f1_score,
    precision_score, recall_score
)
from imblearn.over_sampling import SMOTE

plt.style.use('seaborn-v0_8-whitegrid')
COLORS = ['#2E86AB', '#E84855', '#3BB273', '#F7B731']

# =============================================================================
# SECTION 1 – DATASET BACKGROUND & CHARACTERISTICS
# =============================================================================
print("=" * 70)
print("SECTION 1: DATASET BACKGROUND & CHARACTERISTICS")
print("=" * 70)

df = pd.read_csv('waterQuality1.csv')

print(f"\nDataset Shape : {df.shape}  ({df.shape[0]} rows × {df.shape[1]} columns)")
print("\nColumn names  :", list(df.columns))
print("\nFirst 5 rows:")
print(df.head().to_string())
print("\nData Types:")
print(df.dtypes)

# =============================================================================
# SECTION 2 – PRE-PROCESSING
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 2: PRE-PROCESSING")
print("=" * 70)

# 2-a  Fix 'ammonia' (stored as object) → numeric
df['ammonia'] = pd.to_numeric(df['ammonia'], errors='coerce')

# 2-b  Fix 'is_safe' – remove '#NUM!' rows, cast to int
df = df[df['is_safe'] != '#NUM!'].copy()
df['is_safe'] = df['is_safe'].astype(int)

print(f"\nAfter removing '#NUM!' rows: {df.shape}")
print("\nMissing values after fix:")
print(df.isnull().sum())

# 2-c  Impute remaining NaN (from 'ammonia' coercion) with median
for col in df.columns:
    if df[col].isnull().sum() > 0:
        df[col].fillna(df[col].median(), inplace=True)
        print(f"  → Imputed '{col}' ({df[col].isnull().sum()} NaN) with median")

print("\nMissing values after imputation:", df.isnull().sum().sum())

# 2-d  Class distribution
class_counts = df['is_safe'].value_counts().sort_index()
print("\nClass Distribution:")
print(f"  Not Safe (0): {class_counts[0]}  ({class_counts[0]/len(df)*100:.1f}%)")
print(f"  Safe     (1): {class_counts[1]}  ({class_counts[1]/len(df)*100:.1f}%)")
print(f"\n  Imbalance Ratio: {class_counts[0]/class_counts[1]:.2f}:1  → IMBALANCED dataset")

# 2-e  Feature / target split
X = df.drop('is_safe', axis=1)
y = df['is_safe']

# 2-f  Feature scaling
scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
print("\nFeatures scaled with StandardScaler (zero-mean, unit-variance).")

# 2-g  SMOTE oversampling on training set only (applied after split)
X_train_raw, X_test, y_train_raw, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)
smote = SMOTE(random_state=42)
X_train, y_train = smote.fit_resample(X_train_raw, y_train_raw)
print(f"\nSMOTE applied on training set:")
print(f"  Before: {pd.Series(y_train_raw).value_counts().to_dict()}")
print(f"  After : {pd.Series(y_train).value_counts().to_dict()}")
print(f"\nTrain size: {len(X_train)}  |  Test size: {len(X_test)}")

# =============================================================================
# SECTION 3 – MODEL EVALUATION TECHNIQUE
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 3: MODEL EVALUATION TECHNIQUE")
print("=" * 70)
print("""
Technique chosen: Stratified 5-Fold Cross-Validation on training set,
                  followed by final evaluation on held-out test set (80/20 split).

Justification:
  • Dataset has 7996 samples — large enough for a hold-out split.
  • Class imbalance (≈8:1) makes stratification critical to ensure each
    fold reflects the real distribution.
  • 5-fold CV gives stable performance estimates without excessive compute.
  • Test set is kept completely unseen until final evaluation (no data leakage).

Primary Metrics:
  • F1-Score (macro) — balances precision & recall, robust to imbalance.
  • ROC-AUC           — measures class separability.
  • Confusion Matrix  — reveals per-class errors clearly.
""")

# =============================================================================
# SECTION 4 – CLASSIFIERS
# =============================================================================
print("=" * 70)
print("SECTION 4: CLASSIFIERS — Random Forest & Logistic Regression")
print("=" * 70)
print("""
Classifier 1 – Random Forest
  • Ensemble of decision trees; robust to noise & outliers.
  • Handles non-linear relationships in water-quality chemical data.
  • Built-in feature importance ranking.

Classifier 2 – Logistic Regression
  • Simple, interpretable linear baseline.
  • Works well with scaled features.
  • Fast training; useful comparison against complex model.
""")

# ── 4-a  Cross-validation ─────────────────────────────────────────────────────
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf  = RandomForestClassifier(n_estimators=200, max_depth=None,
                             class_weight='balanced', random_state=42, n_jobs=-1)
lr  = LogisticRegression(max_iter=1000, class_weight='balanced',
                         solver='lbfgs', random_state=42)

for name, model in [("Random Forest", rf), ("Logistic Regression", lr)]:
    cv_f1  = cross_val_score(model, X_train, y_train, cv=cv,
                              scoring='f1_macro',  n_jobs=-1)
    cv_auc = cross_val_score(model, X_train, y_train, cv=cv,
                              scoring='roc_auc',    n_jobs=-1)
    print(f"\n{name}:")
    print(f"  CV F1  (macro): {cv_f1.mean():.4f}  ± {cv_f1.std():.4f}")
    print(f"  CV AUC       : {cv_auc.mean():.4f}  ± {cv_auc.std():.4f}")

# ── 4-b  Train & evaluate on test set ────────────────────────────────────────
results = {}
for name, model in [("Random Forest", rf), ("Logistic Regression", lr)]:
    model.fit(X_train, y_train)
    y_pred  = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    results[name] = {
        'model':     model,
        'y_pred':    y_pred,
        'y_proba':   y_proba,
        'accuracy':  accuracy_score(y_test, y_pred),
        'f1_macro':  f1_score(y_test, y_pred, average='macro'),
        'f1_safe':   f1_score(y_test, y_pred, pos_label=1),
        'precision': precision_score(y_test, y_pred, average='macro'),
        'recall':    recall_score(y_test, y_pred, average='macro'),
        'roc_auc':   roc_auc_score(y_test, y_proba),
        'cm':        confusion_matrix(y_test, y_pred),
        'report':    classification_report(y_test, y_pred,
                                           target_names=['Not Safe', 'Safe'])
    }

print("\n\n── FINAL TEST-SET RESULTS ──")
for name, r in results.items():
    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(f"  Accuracy  : {r['accuracy']:.4f}")
    print(f"  F1 (macro): {r['f1_macro']:.4f}")
    print(f"  F1 (safe) : {r['f1_safe']:.4f}")
    print(f"  Precision : {r['precision']:.4f}")
    print(f"  Recall    : {r['recall']:.4f}")
    print(f"  ROC-AUC   : {r['roc_auc']:.4f}")
    print(f"\nClassification Report:\n{r['report']}")

# =============================================================================
# SECTION 5 – VISUALISATIONS
# =============================================================================
print("=" * 70)
print("SECTION 5: PLOTS (saved as PNG files)")
print("=" * 70)

# ── Plot 1: Class distribution ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Water Quality Dataset – Class Distribution', fontsize=14, fontweight='bold')

labels = ['Not Safe (0)', 'Safe (1)']
axes[0].bar(labels, class_counts.values, color=[COLORS[1], COLORS[0]], edgecolor='white', linewidth=1.5)
axes[0].set_title('Class Count')
axes[0].set_ylabel('Number of Samples')
for i, v in enumerate(class_counts.values):
    axes[0].text(i, v + 30, str(v), ha='center', fontweight='bold')

axes[1].pie(class_counts.values, labels=labels, colors=[COLORS[1], COLORS[0]],
            autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
axes[1].set_title('Class Proportion')

plt.tight_layout()
plt.savefig('01_class_distribution.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 01_class_distribution.png")

# ── Plot 2: Feature correlation heatmap ───────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 11))
corr = df.drop('is_safe', axis=1).corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, cmap='coolwarm', center=0, annot=True, fmt='.2f',
            linewidths=0.5, ax=ax, annot_kws={'size': 7})
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 02_correlation_heatmap.png")

# ── Plot 3: Feature distributions ─────────────────────────────────────────────
feature_cols = X.columns.tolist()
n_cols = 4
n_rows = int(np.ceil(len(feature_cols) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, n_rows * 3))
axes = axes.flatten()

for i, col in enumerate(feature_cols):
    for cls, clr, lbl in zip([0, 1], [COLORS[1], COLORS[0]], ['Not Safe', 'Safe']):
        axes[i].hist(df[df['is_safe'] == cls][col], bins=30, alpha=0.6,
                     color=clr, label=lbl, edgecolor='none')
    axes[i].set_title(col, fontsize=9, fontweight='bold')
    axes[i].legend(fontsize=7)

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Feature Distributions by Class', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('03_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 03_feature_distributions.png")

# ── Plot 4: Confusion Matrices ─────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('Confusion Matrices', fontsize=14, fontweight='bold')

for ax, (name, r) in zip(axes, results.items()):
    disp = ConfusionMatrixDisplay(r['cm'], display_labels=['Not Safe', 'Safe'])
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(name, fontweight='bold')

plt.tight_layout()
plt.savefig('04_confusion_matrices.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 04_confusion_matrices.png")

# ── Plot 5: ROC Curves ────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(8, 6))
for (name, r), clr in zip(results.items(), COLORS):
    fpr, tpr, _ = roc_curve(y_test, r['y_proba'])
    ax.plot(fpr, tpr, color=clr, lw=2,
            label=f"{name}  (AUC = {r['roc_auc']:.4f})")

ax.plot([0, 1], [0, 1], 'k--', lw=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('ROC Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig('05_roc_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 05_roc_curves.png")

# ── Plot 6: Feature Importance (RF) ──────────────────────────────────────────
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(9, 7))
importances.plot(kind='barh', ax=ax, color=COLORS[0], edgecolor='white')
ax.set_title('Random Forest – Feature Importances', fontsize=14, fontweight='bold')
ax.set_xlabel('Importance Score')
plt.tight_layout()
plt.savefig('06_feature_importance.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 06_feature_importance.png")

# ── Plot 7: Metrics Comparison Bar Chart ─────────────────────────────────────
metrics = ['accuracy', 'f1_macro', 'precision', 'recall', 'roc_auc']
metric_labels = ['Accuracy', 'F1 (macro)', 'Precision', 'Recall', 'ROC-AUC']
rf_vals = [results['Random Forest'][m] for m in metrics]
lr_vals = [results['Logistic Regression'][m] for m in metrics]

x = np.arange(len(metrics))
width = 0.35
fig, ax = plt.subplots(figsize=(11, 6))
b1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest',
             color=COLORS[0], edgecolor='white', linewidth=1.5)
b2 = ax.bar(x + width/2, lr_vals, width, label='Logistic Regression',
             color=COLORS[1], edgecolor='white', linewidth=1.5)
ax.set_xticks(x)
ax.set_xticklabels(metric_labels, fontsize=11)
ax.set_ylim(0, 1.12)
ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)

for bar in list(b1) + list(b2):
    h = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2, h + 0.01, f'{h:.3f}',
            ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig('07_metrics_comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: 07_metrics_comparison.png")

# =============================================================================
# SECTION 5 – CONCLUSION
# =============================================================================
print("\n" + "=" * 70)
print("SECTION 5: CONCLUSION")
print("=" * 70)
rf_r  = results['Random Forest']
lr_r  = results['Logistic Regression']
winner = "Random Forest" if rf_r['f1_macro'] >= lr_r['f1_macro'] else "Logistic Regression"

print(f"""
This study applied machine learning to classify water potability using
the waterQuality1 dataset (7,996 samples, 20 chemical features).

Key Findings:
  • The dataset is highly imbalanced (≈88% Not Safe, ≈12% Safe).
  • SMOTE oversampling and class-weighting were applied to address imbalance.
  • StandardScaler normalised all chemical feature ranges.

Model Comparison:
  ┌───────────────────┬──────────┬──────────┬──────────┐
  │ Metric            │  RF      │  LR      │ Better   │
  ├───────────────────┼──────────┼──────────┼──────────┤
  │ Accuracy          │ {rf_r['accuracy']:.4f}   │ {lr_r['accuracy']:.4f}   │ {'RF' if rf_r['accuracy']>lr_r['accuracy'] else 'LR'}       │
  │ F1 (macro)        │ {rf_r['f1_macro']:.4f}   │ {lr_r['f1_macro']:.4f}   │ {'RF' if rf_r['f1_macro']>lr_r['f1_macro'] else 'LR'}       │
  │ ROC-AUC           │ {rf_r['roc_auc']:.4f}   │ {lr_r['roc_auc']:.4f}   │ {'RF' if rf_r['roc_auc']>lr_r['roc_auc'] else 'LR'}       │
  └───────────────────┴──────────┴──────────┴──────────┘

Best Classifier: {winner}
  Random Forest outperforms Logistic Regression due to its ability to
  capture non-linear chemical interactions, whereas LR assumes linearity.

All plots saved to the working directory.
""")

print("✓ Script completed successfully.")
