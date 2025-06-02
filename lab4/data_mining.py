import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

# Ustawienie ziarna losowości dla reprodukowalności
np.random.seed(42)

# Wczytanie danych
data = pd.read_csv('cardiotocography_v2.csv', encoding='utf-8')

# ----------------------------
# 1. Podstawowe statystyki
# ----------------------------
print("=== 1. Basic Statistics ===")
print(data.describe())

# Group by CLASS and count missing values for each feature
missing_per_class = data.groupby('CLASS').apply(lambda x: x.isnull().sum())

# Display the result
print("Missing values per class:")
print(missing_per_class)

# Optional: Visualize missing values per class (heatmap)
plt.figure(figsize=(12, 8))
sns.heatmap(missing_per_class.drop('CLASS', axis=1),
            cmap='YlOrRd',
            annot=True,
            fmt='d',
            linewidths=0.5)
plt.title('Missing Values per Feature and Class')
plt.xlabel('Features')
plt.ylabel('Class')
plt.show()

# Liczba wierszy dla każdej klasy
class_counts = data['CLASS'].value_counts().sort_index()

# Liczba brakujących wartości per klasa i cecha
missing_per_class = data.groupby('CLASS').apply(lambda x: x.isnull().sum())

# Normalizacja: brakujące wartości / liczba wierszy w klasie * 100%
missing_percent_per_class = (missing_per_class.div(class_counts, axis=0) * 100).round(2)

# Oblicz min i max procent brakujących wartości (ignorując 0% jeśli potrzebne)
min_missing = missing_percent_per_class.replace(0,
                                                np.nan).min().min()  # Możesz użyć .min().min() bez replace jeśli chcesz uwzględnić 0%
max_missing = missing_percent_per_class.max().max()

# Wizualizacja z dostosowaną skalą
plt.figure(figsize=(14, 8))
heatmap = sns.heatmap(missing_percent_per_class.drop('CLASS', axis=1),
                      cmap='YlOrRd',
                      annot=True,
                      fmt='.1f',
                      linewidths=0.5,
                      vmin=min_missing,
                      vmax=max_missing,
                      cbar_kws={'label': '% missing values'})
plt.title('Procent brakujących wartości względem liczby wierszy w klasie\n(Skala dostosowana do danych)')
plt.xlabel('Cecha')
plt.ylabel('Klasa (CLASS)')

# Dopasuj kolorystykę dla przypadków z 0% brakujących wartości
if min_missing == 0:
    heatmap.set_facecolor('white')  # Opcjonalnie: zaznacz 0% innym kolorem

plt.show()

# ----------------------------
# 2. Brakujące wartości
# ----------------------------
# Wczytanie danych (zakładając, że `data` jest już załadowane)
# data = pd.read_csv('cardiotocography_v2.csv')

# 1. Obliczenie brakujących wartości
print("\n=== 2. Missing Values ===")
missing_values = data.isnull().sum()
print(missing_values)

# 2. Przygotowanie danych do wizualizacji
print("\n=== 2. Missing Values ===")
missing_values = data.isnull().sum()
print(missing_values)

# Przygotowanie danych do wykresu
missing_df = pd.DataFrame({
    'Feature': missing_values.index,
    'Missing_Count': missing_values.values
}).sort_values('Missing_Count', ascending=False)

# Utworzenie wykresu
plt.figure(figsize=(12, 6))
bars = plt.barh(
    missing_df['Feature'],
    missing_df['Missing_Count'],
    color='#1f77b4',  # Klasyczny niebieski (kolor domyślny matplotlib)
    edgecolor='black',  # Czarna obwódka słupków
    linewidth=0.7
)

# Dodanie etykiet z wartościami
for bar in bars:
    width = bar.get_width()
    plt.text(
        width + 0.5,
        bar.get_y() + bar.get_height() / 2,
        f'{int(width)}',
        ha='left',
        va='center',
        fontsize=10
    )

# Dostosowanie osi i siatki
plt.xlim(0, missing_df['Missing_Count'].max() * 1.15)
plt.grid(axis='x', linestyle='--', alpha=0.4)

# Tytuły i etykiety
plt.title('Liczba brakujących wartości dla każdej cechy', pad=20, fontsize=14)
plt.xlabel('Liczba brakujących rekordów', fontsize=12)
plt.ylabel('Cecha', fontsize=12)

plt.tight_layout()
plt.show()

# Wizualizacja brakujących wartości (jeśli istnieją)
if missing_values.any():
    plt.figure(figsize=(10, 5))
    sns.heatmap(data.isnull(), cbar=False, cmap='viridis')
    plt.title('Missing Values Heatmap')
    plt.show()

# ----------------------------
# 3. Rozkład klas (target)
# ----------------------------
print("\n=== 3. Class Distribution ===")
class_distribution = data['CLASS'].value_counts().sort_index()
print(class_distribution)

# Wykres słupkowy rozkładu klas
plt.figure(figsize=(10, 5))
class_distribution.plot(kind='bar', color='skyblue')
plt.title('Class Distribution (Target Variable)')
plt.xlabel('Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

# ----------------------------
# 4. Rozkłady cech (histogramy)
# ----------------------------
print("\n=== 4. Feature Distributions ===")
numeric_features = data.select_dtypes(include=[np.number]).columns.drop('CLASS')

plt.figure(figsize=(15, 20))
for i, col in enumerate(numeric_features, 1):
    plt.subplot(7, 3, i)
    sns.histplot(data[col], kde=True, bins=30, color='teal')
    plt.title(f'Distribution of {col}')
    plt.tight_layout()
plt.show()

# ----------------------------
# 5. Wykresy pudełkowe (boxploty)
# ----------------------------
print("\n=== 5. Boxplots for Features ===")
plt.figure(figsize=(15, 10))
sns.boxplot(data=data[numeric_features], orient='h', palette='Set2')
plt.title('Boxplot of Features')
plt.tight_layout()
plt.show()

# ----------------------------
# 6. Macierz korelacji
# ----------------------------
print("\n=== 6. Correlation Matrix ===")
corr_matrix = data[numeric_features].corr()

plt.figure(figsize=(15, 12))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap='coolwarm',
            annot_kws={'size': 8}, linewidths=0.5)
plt.title('Correlation Matrix of Features')
plt.tight_layout()
plt.show()

# Podsumowanie silnych korelacji (> 0.7 lub < -0.7)
strong_correlations = corr_matrix.unstack().sort_values(ascending=False)
strong_correlations = strong_correlations[(strong_correlations.abs() > 0.7) & (strong_correlations < 1)]
print("\nStrong Correlations (|r| > 0.7):")
print(strong_correlations)

# Wczytanie danych
data = pd.read_csv('cardiotocography_v2.csv', encoding='utf-8')

# Przygotowanie danych - usunięcie brakujących wartości i standaryzacja
data_clean = data.dropna()
X = data_clean.drop('CLASS', axis=1)
y = data_clean['CLASS']

# Standaryzacja danych
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Redukcja wymiarowości za pomocą PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Redukcja wymiarowości za pomocą t-SNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

# Wykresy
plt.figure(figsize=(18, 6))

# PCA
plt.subplot(1, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('PCA - Rozkład klas w 2D')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(scatter, label='Klasa')

# t-SNE
plt.subplot(1, 2, 2)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('t-SNE - Rozkład klas w 2D')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(scatter, label='Klasa')

plt.tight_layout()
plt.show()
# Klasteryzacja K-means
kmeans = KMeans(n_clusters=20, random_state=42)
clusters = kmeans.fit_predict(X_scaled)

# Wykresy z klasteryzacją
plt.figure(figsize=(18, 12))

# PCA z prawdziwymi klasami
plt.subplot(2, 2, 1)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('PCA - Prawdziwe klasy')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(scatter, label='Klasa')

# PCA z klastrami
plt.subplot(2, 2, 2)
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('PCA - Klastry K-means')
plt.xlabel('PCA 1')
plt.ylabel('PCA 2')
plt.colorbar(scatter, label='Klaster')

# t-SNE z prawdziwymi klasami
plt.subplot(2, 2, 3)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.title('t-SNE - Prawdziwe klasy')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(scatter, label='Klasa')

# t-SNE z klastrami
plt.subplot(2, 2, 4)
scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=clusters, cmap='viridis', alpha=0.6)
plt.title('t-SNE - Klastry K-means')
plt.xlabel('t-SNE 1')
plt.ylabel('t-SNE 2')
plt.colorbar(scatter, label='Klaster')

plt.tight_layout()
plt.show()
