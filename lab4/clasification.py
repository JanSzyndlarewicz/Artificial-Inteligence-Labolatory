import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.impute import KNNImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE
from itertools import product
from collections import defaultdict


class DataPreprocessor:
    """Handles all data loading, cleaning, and preprocessing"""

    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_val = None
        self.y_train = None
        self.y_val = None

    def load_data(self):
        """Load and inspect the raw data"""
        try:
            # Try UTF-8 first
            self.data = pd.read_csv(self.file_path, encoding='utf-8')
        except UnicodeDecodeError:
            try:
                # Try latin-1 if UTF-8 fails
                self.data = pd.read_csv(self.file_path, encoding='latin-1')
            except Exception as e:
                print(f"Failed to load file: {e}")
                raise

        print("=== 1. Data Mining ===")
        print("Sample data:\n", self.data.head())
        print("\nMissing values before imputation:\n", self.data.isnull().sum())

    def split_features_target(self, target_col='CLASS'):
        """Split data into features and target"""
        self.X = self.data.drop(target_col, axis=1)
        self.y = self.data[target_col]

    def impute_missing_values(self, n_neighbors=5):
        """Handle missing values using KNN imputation"""
        # Temporary standardization for KNN
        temp_scaler = StandardScaler()
        X_scaled = temp_scaler.fit_transform(self.X)

        # KNN imputation
        knn_imputer = KNNImputer(n_neighbors=n_neighbors)
        X_imputed = knn_imputer.fit_transform(X_scaled)
        X_imputed = temp_scaler.inverse_transform(X_imputed)
        self.X = pd.DataFrame(X_imputed, columns=self.X.columns)

        print("\nMissing values after KNN imputation:\n", self.X.isnull().sum())

    def split_data(self, test_size=0.2, random_state=50, stratify=True):
        """Split data into train and validation sets"""
        stratify = self.y if stratify else None
        self.X_train, self.X_val, self.y_train, self.y_val = train_test_split(
            self.X, self.y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )

    def balance_data(self, random_state=42):
        """Balance classes using SMOTE"""
        smote = SMOTE(random_state=random_state)
        self.X, self.y = smote.fit_resample(self.X, self.y)

    def get_data_versions(self):
        """Generate different preprocessing versions of the data"""
        data_versions = {
            'only_imputed': (self.X_train.copy(), self.X_val.copy()),
        }

        # Standardization
        scaler_std = StandardScaler()
        X_train_std = scaler_std.fit_transform(self.X_train)
        X_val_std = scaler_std.transform(self.X_val)
        data_versions['standardized'] = (X_train_std, X_val_std)

        # Normalization
        scaler_norm = MinMaxScaler()
        X_train_norm = scaler_norm.fit_transform(self.X_train)
        X_val_norm = scaler_norm.transform(self.X_val)
        data_versions['normalized'] = (X_train_norm, X_val_norm)

        # PCA (on standardized data)
        pca = PCA(n_components=0.95)
        X_train_pca = pca.fit_transform(X_train_std)
        X_val_pca = pca.transform(X_val_std)
        data_versions['pca'] = (X_train_pca, X_val_pca)

        return data_versions


class ModelEvaluator:
    """Handles model training, evaluation, and comparison"""

    def __init__(self, X_train, X_val, y_train, y_val):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train = y_train
        self.y_val = y_val
        self.results = []
        self.metrics = {
            'accuracy': accuracy_score,
            'precision': lambda y_true, y_pred: precision_score(y_true, y_pred, average='weighted', zero_division=1),
            'recall': lambda y_true, y_pred: recall_score(y_true, y_pred, average='weighted'),
            'f1': lambda y_true, y_pred: f1_score(y_true, y_pred, average='weighted')
        }

    def get_models(self):
        """Define models and their parameter grids for testing"""
        models = {
            # 'Naive Bayes': {
            #     'model': GaussianNB(),
            #     'params': {}
            # },
            # 'Decision Tree': {
            #     'model': DecisionTreeClassifier(),
            #     'params': {
            #         'max_depth': [None, 3, 4, 5, 10],
            #         'min_samples_split': [2, 5, 10],
            #         'criterion': ['gini', 'entropy']
            #     }
            # },
            # 'Random Forest': {
            #     'model': RandomForestClassifier(),
            #     'params': {
            #         'n_estimators': [100, 200],
            #         'criterion': ['gini', 'entropy'],
            #         'class_weight': [None, 'balanced']
            #     }
            # },
            'SVM': {
                'model': SVC(),
                'params': {
                    'kernel': ['rbf', 'linear'],
                    'C': [0.1, 1, 10, 100]
                }
            }
        }
        return models

    def evaluate_model(self, model, X_train, X_val, y_train, y_val, model_name, data_prep):
        """Train and evaluate a single model"""
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        # Calculate metrics
        metrics = {name: metric(y_val, y_pred) for name, metric in self.metrics.items()}

        # Store results
        result = {
            'Model': model_name,
            'Data Preparation': data_prep,
            **metrics
        }

        # Add model parameters if available
        if hasattr(model, 'get_params'):
            result.update({f'param_{k}': v for k, v in model.get_params().items()})

        self.results.append(result)

        # Plot confusion matrix
        self.plot_confusion_matrix(y_val, y_pred, f"{model_name} with {data_prep}")

        return metrics

    def plot_confusion_matrix(self, y_true, y_pred, title):
        """Plot confusion matrix for model evaluation"""
        plt.figure(figsize=(10, 8))
        cm = confusion_matrix(y_true, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title(f'Confusion Matrix - {title}')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.show()

    def run_experiments(self, data_versions, models=None):
        """Run experiments with all combinations of models and data versions"""
        if models is None:
            models = self.get_models()

        for model_name, model_info in models.items():
            print(f"\nEvaluating {model_name}...")

            # If no parameters to test, just use the base model
            if not model_info['params']:
                for data_name, (X_tr, X_v) in data_versions.items():
                    self.evaluate_model(
                        model_info['model'],
                        X_tr, X_v,
                        self.y_train, self.y_val,
                        model_name, data_name
                    )
                continue

            # Generate all parameter combinations
            keys, values = zip(*model_info['params'].items())
            for params in product(*values):
                params_dict = dict(zip(keys, params))

                # Create model with current parameters
                current_model = model_info['model'].set_params(**params_dict)

                # Test with all data versions
                for data_name, (X_tr, X_v) in data_versions.items():
                    self.evaluate_model(
                        current_model,
                        X_tr, X_v,
                        self.y_train, self.y_val,
                        model_name, data_name
                    )

    def get_results_df(self):
        """Return results as a DataFrame"""
        return pd.DataFrame(self.results)

    def show_top_results(self, n=10, metric='accuracy'):
        """Display top performing models"""
        df = self.get_results_df()
        return df.sort_values(by=metric, ascending=False).head(n)

    def cross_validate_best_model(self, X, y, model, cv=5):
        """Perform cross-validation on the best model"""
        print(f"\nCross-validation for best model ({model.__class__.__name__}):")
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X, y, cv=kf, scoring='accuracy')
        print(f"Cross-validation scores: {cv_scores}")
        print(f"Mean CV accuracy: {np.mean(cv_scores):.3f}")
        return cv_scores

    def plot_feature_importance(self, model, feature_names):
        """Plot feature importance for tree-based models"""
        if hasattr(model, 'feature_importances_'):
            plt.figure(figsize=(12, 8))
            importances = model.feature_importances_
            indices = np.argsort(importances)[::-1]
            plt.title('Feature Importances')
            plt.bar(range(len(feature_names)), importances[indices], align='center')
            plt.xticks(range(len(feature_names)), [feature_names[i] for i in indices], rotation=90)
            plt.tight_layout()
            plt.show()


def main():
    # Initialize data preprocessor
    preprocessor = DataPreprocessor('cardiotocography_v2.csv')
    preprocessor.load_data()
    preprocessor.split_features_target()
    preprocessor.impute_missing_values()

    # Choose whether to balance data or not
    USE_BALANCED_DATA = True

    if USE_BALANCED_DATA:
        preprocessor.balance_data()
        preprocessor.split_data(stratify=False)  # No need to stratify after balancing
    else:
        preprocessor.split_data()

    # Get different data processing versions
    data_versions = preprocessor.get_data_versions()

    # Initialize evaluator with the prepared data
    evaluator = ModelEvaluator(
        preprocessor.X_train,
        preprocessor.X_val,
        preprocessor.y_train,
        preprocessor.y_val
    )

    # Run experiments with all combinations
    evaluator.run_experiments(data_versions)

    # Show results
    results_df = evaluator.get_results_df()
    print("\n=== Results Summary ===")
    print(results_df.to_markdown(tablefmt="grid", floatfmt=".3f"))

    print("\n=== Top 10 Models by Accuracy ===")
    print(evaluator.show_top_results(10, 'accuracy').to_markdown(tablefmt="grid", floatfmt=".3f"))

    # Further analysis on best model
    best_model_info = evaluator.show_top_results(1, 'accuracy').iloc[0]
    print(f"\nBest model: {best_model_info['Model']} with {best_model_info['Data Preparation']}")
    print(f"Parameters: { {k: v for k, v in best_model_info.items() if k.startswith('param_')} }")

    # Reinstantiate best model with its parameters
    models = evaluator.get_models()
    best_model = models[best_model_info['Model']]['model']
    best_model.set_params(**{k.replace('param_', ''): v
                             for k, v in best_model_info.items()
                             if k.startswith('param_')})

    # Cross-validation
    evaluator.cross_validate_best_model(preprocessor.X, preprocessor.y, best_model)

    # Feature importance (if applicable)
    if best_model_info['Model'] in ['Decision Tree', 'Random Forest']:
        evaluator.plot_feature_importance(best_model, preprocessor.X.columns)


if __name__ == "__main__":
    main()