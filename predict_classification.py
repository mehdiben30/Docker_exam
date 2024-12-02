import os  # Import os module to handle file paths
import joblib
import numpy as np
from sklearn.metrics import accuracy_score, classification_report

# Charger les données de validation et le nom du modèle
def load_validation_data():
    try:
        # Déterminer le répertoire du script actuel
        try:
            script_path = os.path.abspath(__file__)  # Chemin absolu du script
            script_dir = os.path.dirname(script_path)  # Répertoire contenant le script
        except NameError:
            # Si __file__ n'est pas défini (par exemple, dans un environnement interactif)
            script_dir = os.getcwd()  # Utiliser le répertoire de travail actuel

        # Créer le chemin complet pour 'validation_data.npz'
        validation_path = os.path.join(script_dir, "validation_data.npz")

        # Charger les données de validation
        data = np.load(validation_path)
        X_val = data["X_val"]
        y_val = data["y_val"]
        model_name = data["model_name"].item()  # Récupérer le nom du modèle

        print(f"Modèle à évaluer : {model_name}")
        return X_val, y_val, model_name, script_dir
    except FileNotFoundError:
        print("Les données de validation ou le modèle choisi n'ont pas été trouvés. Assurez-vous d'avoir exécuté l'étape d'entraînement.")
        exit()
    except KeyError as e:
        print(f"Clé manquante dans 'validation_data.npz' : {e}")
        exit()
    except Exception as e:
        print(f"Erreur lors du chargement des données de validation : {e}")
        exit()

# Charger le modèle optimisé correspondant
def load_trained_model(model_name, script_dir):
    try:
        # Créer le chemin complet pour le modèle optimisé
        model_filename = f"{model_name}_optimized_model.pkl"
        model_path = os.path.join(script_dir, model_filename)

        # Charger le modèle optimisé
        trained_model = joblib.load(model_path)
        print(f"Modèle '{model_name}' chargé avec succès depuis : {model_path}")
        return trained_model
    except FileNotFoundError:
        print(f"Le modèle optimisé '{model_name}_optimized_model.pkl' n'a pas été trouvé dans {script_dir}. Assurez-vous de l'avoir sauvegardé après l'optimisation.")
        exit()
    except Exception as e:
        print(f"Erreur lors du chargement du modèle optimisé : {e}")
        exit()

# Faire des prédictions sur les données de validation et évaluer les performances
def evaluate_model(trained_model, X_val, y_val, model_name):
    try:
        # Faire des prédictions sur les données de validation
        y_pred = trained_model.predict(X_val)

        # Évaluer les performances
        acc = accuracy_score(y_val, y_pred)
        print(f"\nAccuracy sur les données de validation avec '{model_name}' : {acc:.2f}")
        print("\nRapport de classification :")
        print(classification_report(y_val, y_pred))
    except Exception as e:
        print(f"Erreur lors de l'évaluation du modèle : {e}")
        exit()

if __name__ == "__main__":
    # Charger les données de validation et le nom du modèle
    X_val, y_val, model_name, script_dir = load_validation_data()

    # Charger le modèle optimisé correspondant
    trained_model = load_trained_model(model_name, script_dir)

    # Évaluer le modèle sur les données de validation
    evaluate_model(trained_model, X_val, y_val, model_name)
