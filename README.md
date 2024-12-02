# Docker_exam

## train_classifier.py
Ce script permet de générer des données synthétiques, de sélectionner et optimiser un modèle de classification, puis de sauvegarder le modèle entraîné et un jeu de données de validation. Voici les principales étapes :

Génération des données : Crée des données de classification synthétiques avec make_classification, puis les divise en jeux d'entraînement, de test et de validation.

Optimisation des modèles : Utilise GridSearchCV pour trouver les meilleurs hyperparamètres parmi plusieurs modèles (Random Forest, Régression Logistique, SVM).

Entraînement et évaluation : Entraîne le modèle optimisé sur les données d'entraînement, puis évalue ses performances sur le jeu de test en affichant l'accuracy et un rapport de classification.

Sauvegarde :
Sauvegarde le modèle optimisé sous forme de fichier .pkl.
Sauvegarde les données de validation et le nom du modèle dans un fichier .npz.

## predict_classification.py
Ce script utilise un modèle préalablement entraîné et sauvegardé (via train_classifier.py) pour évaluer ses performances sur le jeu de données de validation. Voici les étapes principales :

Chargement des données : Charge les données de validation et le nom du modèle sauvegardés dans le fichier .npz.

Chargement du modèle : Récupère le modèle optimisé sauvegardé sous forme de fichier .pkl.

Évaluation du modèle : Fait des prédictions sur les données de validation, calcule l'accuracy et affiche un rapport de classification détaillé.

## Utilisation
Étape 1 : Entraînement
Exécutez train_classifier.py pour générer les données, optimiser et entraîner un modèle, puis sauvegarder le modèle et les données de validation.

Étape 2 : Prédictions et évaluation
Exécutez predict_classification.py pour évaluer les performances du modèle sauvegardé sur les données de validation.
