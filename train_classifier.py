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

# try to create a conflict By Bilal

# conflict resolved ;)

# Entraîner et évaluer le modèle
def train_and_evaluate(model, X_train, y_train, X_test, y_test):
    print("\nÉvaluation après optimisation :")
    model.fit(X_train, y_train)  # Entraînement
    y_pred = model.predict(X_test)  # Prédictions sur les données de test
    acc = accuracy_score(y_test, y_pred)  # Calcul de l'accuracy
    print(f"\nModèle : {model.__class__.__name__}")
    print(f"Accuracy sur le test : {acc:.2f}")
    print("\nRapport de classification :")
    print(classification_report(y_test, y_pred))
    return model

if __name__ == "__main__":
    print("Génération des données...\n")
    X_train, X_test, X_val, y_train, y_test, y_val = generate_data()

    models = {
        "RandomForest": RandomForestClassifier(random_state=42),
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "SVM": SVC(random_state=42)
    }

    param_grids = {
        "RandomForest": {
            "n_estimators": [50, 100, 200],
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5, 10]
        },
        "LogisticRegression": {
            "C": [0.1, 1, 10],
            "penalty": ["l2"]
        },
        "SVM": {
            "C": [0.1, 1, 10],
            "kernel": ["linear", "rbf"]
        }
    }

    print("Choisissez un modèle à optimiser :")
    print(", ".join(models.keys()))
    model_name = input("Votre choix : ").strip()

    if model_name in models:
        # Optimiser le modèle
        best_model = optimize_model(models[model_name], param_grids[model_name], X_train, y_train)

        # Entraîner et évaluer le modèle optimisé
        trained_model = train_and_evaluate(best_model, X_train, y_train, X_test, y_test)

        # Déterminer le répertoire du script actuel
        try:
            script_path = os.path.abspath(__file__)  # Chemin absolu du script
            script_dir = os.path.dirname(script_path)  # Répertoire contenant le script
        except NameError:
            # Si __file__ n'est pas défini (par exemple, dans un environnement interactif)
            script_dir = os.getcwd()  # Utiliser le répertoire de travail actuel

        # Créer les chemins complets pour les fichiers à sauvegarder
        model_filename = f"{model_name}_optimized_model.pkl"
        validation_filename = "validation_data.npz"

        model_path = os.path.join(script_dir, model_filename)
        validation_path = os.path.join(script_dir, validation_filename)

        # Sauvegarder le modèle optimisé
        try:
            joblib.dump(trained_model, model_path)
            print(f"Modèle sauvegardé à : {model_path}")
        except Exception as e:
            print(f"Échec de la sauvegarde du modèle : {e}")

        # Sauvegarder les données de validation et le nom du modèle
        try:
            np.savez(validation_path, X_val=X_val, y_val=y_val, model_name=model_name)
            print(f"Données de validation sauvegardées à : {validation_path}")
        except Exception as e:
            print(f"Échec de la sauvegarde des données de validation : {e}")
    else:
        print(f"Modèle '{model_name}' non reconnu. Veuillez choisir parmi : {', '.join(models.keys())}.")
