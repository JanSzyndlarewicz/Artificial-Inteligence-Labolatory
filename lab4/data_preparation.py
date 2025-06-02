import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA

# 1. Load data
data = pd.read_csv('cardiotocography_v2.csv')
X = data.drop('CLASS', axis=1)
y = data['CLASS']

# 2. KNN Imputation
print("\n=== Before imputation (first 5 rows) ===")
print(X.head().to_markdown(tablefmt="grid", floatfmt=".2f"))

imputer = KNNImputer(n_neighbors=5)
X_imputed = imputer.fit_transform(X)
X_imputed = pd.DataFrame(X_imputed, columns=X.columns)

print("\n=== After KNN imputation (first 5 rows) ===")
print(X_imputed.head().to_markdown(tablefmt="grid", floatfmt=".2f"))

# 3. Standardization and Normalization
std_scaler = StandardScaler()
X_std = std_scaler.fit_transform(X_imputed)
X_std = pd.DataFrame(X_std, columns=X.columns)

minmax_scaler = MinMaxScaler()
X_norm = minmax_scaler.fit_transform(X_imputed)
X_norm = pd.DataFrame(X_norm, columns=X.columns)

# 4. PCA on standardized data
pca = PCA(n_components=0.95)
X_pca = pca.fit_transform(X_std)

# 5. Visualization of processing steps
plt.figure(figsize=(18, 12))

# Original data
plt.subplot(2, 3, 1)
plt.boxplot(X.values)
plt.title('Original Data')
plt.xticks(ticks=range(1, len(X.columns) + 1), labels=X.columns, rotation=90)

# After imputation
plt.subplot(2, 3, 2)
plt.boxplot(X_imputed.values)
plt.title('After KNN Imputation')
plt.xticks(ticks=range(1, len(X.columns) + 1), labels=X.columns, rotation=90)

# After standardization
plt.subplot(2, 3, 3)
plt.boxplot(X_std.values)
plt.title('After Standardization')
plt.xticks(ticks=range(1, len(X.columns) + 1), labels=X.columns, rotation=90)

# After normalization
plt.subplot(2, 3, 4)
plt.boxplot(X_norm.values)
plt.title('After Normalization (MinMax)')
plt.xticks(ticks=range(1, len(X.columns) + 1), labels=X.columns, rotation=90)

# After PCA
plt.subplot(2, 3, 5)
plt.boxplot(X_pca)
plt.title(f'After PCA ({X_pca.shape[1]} components)')
plt.xticks(ticks=range(1, X_pca.shape[1] + 1), labels=[f"PC{i}" for i in range(1, X_pca.shape[1] + 1)], rotation=90)

plt.tight_layout()
plt.show()

# 6. Feature distribution comparison
features_to_compare = ['LB', 'MSTV', 'ASTV']
plt.figure(figsize=(18, 6))

colors = {
    'Original': '#3498db',  # blue
    'After KNN': '#e74c3c',  # red
    'Standardized': '#2ecc71',  # green
    'Normalized': '#9b59b6'  # purple
}

alphas = {
    'Original': 0.7,
    'After KNN': 0.7,
    'Standardized': 0.7,
    'Normalized': 0.7
}

for i, feature in enumerate(features_to_compare, 1):
    plt.subplot(1, 3, i)

    # Plot histograms
    plt.hist(X_imputed[feature], bins=30, color=colors['After KNN'],
             alpha=alphas['After KNN'], label='After KNN')
    plt.hist(X_std[feature], bins=30, color=colors['Standardized'],
             alpha=alphas['Standardized'], label='Standardized')
    plt.hist(X_norm[feature], bins=30, color=colors['Normalized'],
             alpha=alphas['Normalized'], label='Normalized')
    plt.hist(X[feature], bins=30, color=colors['Original'],
             alpha=alphas['Original'], label='Original')

    plt.title(f'Distribution of {feature}', fontsize=12, pad=12)
    plt.xlabel('Value', fontsize=10)
    plt.ylabel('Frequency', fontsize=10)

    plt.legend(framealpha=0.9, fontsize=9,
               facecolor='white', edgecolor='gray',
               bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.grid(axis='y', linestyle=':', alpha=0.4)

    # Set consistent x-axis limits
    all_versions = [X[feature], X_imputed[feature], X_std[feature], X_norm[feature]]
    x_min = min(v.min() for v in all_versions) * 0.95
    x_max = max(v.max() for v in all_versions) * 1.05
    plt.xlim(x_min, x_max)

plt.suptitle('Feature Distributions at Different Processing Stages',
             fontsize=14, y=1.05)
plt.tight_layout()
plt.show()