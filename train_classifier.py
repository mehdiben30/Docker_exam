import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import seaborn as sns
import joblib

# Générer les données
def generate_data(n_samples=1000, n_features=20, n_informative=5, n_classes=3, random_state=42):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=random_state
    )
    # Séparer les données : 70% entraînement, 20% test, 10% validation
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=random_state)
    X_test, X_val, y_test, y_val = train_test_split(X_temp, y_temp, test_size=0.33, random_state=random_state)
    return X_train, X_test, X_val, y_train, y_test, y_val


# Cross-Validation et Grid Search
def optimize_model(model, param_grid, X_train, y_train):
    print(f"Optimisation du modèle {model.__class__.__name__} avec Grid Search...\n")
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    print(f"Meilleurs paramètres pour {model.__class__.__name__} : {grid_search.best_params_}")
    print(f"Meilleure score de cross-validation : {grid_search.best_score_:.2f}")
    return grid_search.best_estimator_
# creating conflict (khalil)
