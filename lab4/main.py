import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import KNNImputer
from sklearn.metrics import (
    f1_score, recall_score,
    precision_score, accuracy_score
)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB, GaussianNB, CategoricalNB
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

# 1. Wczytanie i imputacja braków
df = pd.read_csv('cardiotocography_v2.csv')
X = df.drop('CLASS', axis=1)
y = df['CLASS']

imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 2. Podział na zbiór treningowy i walidacyjny
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 3. Standaryzacja
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train)
# X_val = scaler.transform(X_val)

scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

median = np.median(X_train)
mean = np.mean(X_train)

results = []


def evaluate_model(model, model_name, param_name, param_value):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_val)
    results.append({
        'Model': model_name,
        'Param': param_name,
        'Value': param_value,
        'F1': f1_score(y_val, y_pred, average='weighted', zero_division=1),
        'Recall': recall_score(y_val, y_pred, average='weighted', zero_division=1),
        'Precision': precision_score(y_val, y_pred, average='weighted', zero_division=1),
        'Accuracy': accuracy_score(y_val, y_pred)
    })


# === BernoulliNB ===
# hiperparametry: alpha, binarize, class_prior

evaluate_model(BernoulliNB(alpha=0.5), 'BernoulliNB', 'alpha', 0.5)
evaluate_model(BernoulliNB(alpha=1.0), 'BernoulliNB', 'alpha', 1.0)
evaluate_model(BernoulliNB(alpha=1.5), 'BernoulliNB', 'alpha', 1.5)


evaluate_model(GaussianNB(var_smoothing=1e-9), 'GaussianNB', 'var_smoothing', 1e-9)
evaluate_model(GaussianNB(var_smoothing=1e-5), 'GaussianNB', 'var_smoothing', 1e-5)
evaluate_model(GaussianNB(var_smoothing=1e-3), 'GaussianNB', 'var_smoothing', 1e-3)

evaluate_model(CategoricalNB(min_categories=2), 'CategoricalNB', 'min_categories', 2)
evaluate_model(CategoricalNB(min_categories=5), 'CategoricalNB', 'min_categories', 5)
evaluate_model(CategoricalNB(min_categories=10), 'CategoricalNB', 'min_categories', 10)

# === DecisionTree ===
# hiperparametry: max_depth, min_samples_split, criterion

evaluate_model(DecisionTreeClassifier(max_depth=2, random_state=42),
               'DecisionTree', 'max_depth', 2)
evaluate_model(DecisionTreeClassifier(max_depth=5, random_state=42),
               'DecisionTree', 'max_depth', 5)
evaluate_model(DecisionTreeClassifier(max_depth=10, random_state=42),
               'DecisionTree', 'max_depth', 10)

evaluate_model(DecisionTreeClassifier(min_samples_split=2, random_state=42),
               'DecisionTree', 'min_samples_split', 2)
evaluate_model(DecisionTreeClassifier(min_samples_split=5, random_state=42),
               'DecisionTree', 'min_samples_split', 5)
evaluate_model(DecisionTreeClassifier(min_samples_split=10, random_state=42),
               'DecisionTree', 'min_samples_split', 10)

evaluate_model(DecisionTreeClassifier(criterion='gini', random_state=42),
               'DecisionTree', 'criterion', 'gini')
evaluate_model(DecisionTreeClassifier(criterion='entropy', random_state=42),
               'DecisionTree', 'criterion', 'entropy')
evaluate_model(DecisionTreeClassifier(criterion='log_loss', random_state=42),
               'DecisionTree', 'criterion', 'log_loss')

# === RandomForest ===
# hiperparametry: n_estimators, max_depth, max_features

evaluate_model(RandomForestClassifier(n_estimators=2, random_state=42),
               'RandomForest', 'n_estimators', 2)
evaluate_model(RandomForestClassifier(n_estimators=5, random_state=42),
               'RandomForest', 'n_estimators', 5)
evaluate_model(RandomForestClassifier(n_estimators=10, random_state=42),
               'RandomForest', 'n_estimators', 10)

evaluate_model(RandomForestClassifier(max_depth=2, random_state=42),
               'RandomForest', 'max_depth', 2)
evaluate_model(RandomForestClassifier(max_depth=5, random_state=42),
               'RandomForest', 'max_depth', 5)
evaluate_model(RandomForestClassifier(max_depth=10, random_state=42),
               'RandomForest', 'max_depth', 10)

evaluate_model(RandomForestClassifier(max_features='sqrt', random_state=42),
               'RandomForest', 'max_features', 'sqrt')
evaluate_model(RandomForestClassifier(max_features='log2', random_state=42),
               'RandomForest', 'max_features', 'log2')
evaluate_model(RandomForestClassifier(max_features=None, random_state=42),
               'RandomForest', 'max_features', 'None')


# 5. Podsumowanie wyników
df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))



def eval_tree(model, name):
    model.fit(X_train, y_train)
    # predykcja na treningu
    y_tr = model.predict(X_train)
    # predykcja na teście
    y_te = model.predict(X_val)
    return [
        {
            'Model': name,
            'Split': 'train',
            'F1':        f1_score(y_train,  y_tr, average='weighted'),
            'Recall':    recall_score(y_train,  y_tr, average='weighted'),
            'Precision': precision_score(y_train,  y_tr, average='weighted'),
            'Accuracy':  accuracy_score(y_train,  y_tr),
        },
        {
            'Model': name,
            'Split': 'test',
            'F1':        f1_score(y_val,     y_te, average='weighted'),
            'Recall':    recall_score(y_val,     y_te, average='weighted'),
            'Precision': precision_score(y_val,     y_te, average='weighted'),
            'Accuracy':  accuracy_score(y_val,     y_te),
        }
    ]



print("\n=== Overfitting reduction - Decision Tree Evaluation ===")

res_unpruned = eval_tree(
    DecisionTreeClassifier(random_state=42),
    'unpruned'
)

res_pruned = eval_tree(
    DecisionTreeClassifier(max_depth=5, random_state=42),
    'max_depth=5'
)

df_cmp = pd.DataFrame(res_unpruned + res_pruned)
print(df_cmp.to_markdown(index=False))






# 1. Load & impute
df = pd.read_csv('cardiotocography_v2.csv')
X = df.drop('CLASS', axis=1)
y = df['CLASS']

imputer = KNNImputer(n_neighbors=5)
X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

# 2. Split into training & validation
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Prepare three versions of the data
# 3a. Raw (no processing)
Xr_train, Xr_val = X_train, X_val

# 3b. Standardization
scaler = StandardScaler().fit(X_train)
Xs_train = scaler.transform(X_train)
Xs_val   = scaler.transform(X_val)

# 3c. PCA (after standardization)
pca = PCA(n_components=5, random_state=42).fit(Xs_train)
Xp_train = pca.transform(Xs_train)
Xp_val   = pca.transform(Xs_val)

# 4. Evaluation helper
results = []
def evaluate_model(model, proc_tag, X_tr, X_te):
    model.fit(X_tr, y_train)  # Trenuj na danych przetworzonych
    ys_pred = model.predict(X_te)  # Przewiduj na danych walidacyjnych
    results.append({
        'Processing': proc_tag,
        'F1': f1_score(y_val, ys_pred, average='weighted', zero_division=1),
        'Recall': recall_score(y_val, ys_pred, average='weighted', zero_division=1),
        'Precision': precision_score(y_val, ys_pred, average='weighted', zero_division=1),
        'Accuracy': accuracy_score(y_val, ys_pred)
    })


# 5. Run experiments with Decision Tree (default settings)
clf = DecisionTreeClassifier(random_state=42)

# Wywołanie funkcji z poprawionymi danymi
evaluate_model(clf, 'raw (Decision Tree)', Xr_train, Xr_val)  # Surowe dane
evaluate_model(clf, 'standardization (Decision Tree)', Xs_train, Xs_val)  # Standaryzacja
evaluate_model(clf, 'PCA (Decision Tree)', Xp_train, Xp_val)  # PCA

# === GaussianNB na tych samych wersjach danych ===
gnb = GaussianNB()
evaluate_model(gnb, 'raw (GaussianNB)', Xr_train, Xr_val)
evaluate_model(gnb, 'standardization (GaussianNB)', Xs_train, Xs_val)
evaluate_model(gnb, 'PCA (GaussianNB)', Xp_train, Xp_val)

# 6. Summarize results

print("\n=== Results of Decision Tree and GaussianNB with different preprocessing ===")

df_results = pd.DataFrame(results)
print(df_results.to_markdown(index=False))