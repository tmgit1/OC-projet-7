🐕 Dog Breed Classification - Comparaison de Modèles de Deep Learning
Ce projet est une extension du projet 5 de classification d'images de races de chiens, dans lequel était utilisé le modèle MobileNetV1. Dans cette nouvelle version, un modèle de deep learning plus récent a été utilisé, le modèle EfficientNetV2B0, afin d'améliorer les performances en termes de précision et de temps de traitement. Un tableau de bord interactif a également été créé avec Streamlit, permettant d'explorer le jeu de données et de comparer les deux modèles sur plusieurs métriques.

📋 Objectifs du Projet
Entraîner un modèle plus récent (EfficientNetV2B0) puis comparer ses performances avec le modèle plus ancien (MobileNetV1).
Offrir aux utilisateurs un tableau de bord pour visualiser les résultats, explorer le jeu de données et tester le nouveau modèle en téléchargeant leurs propres images.
Mesurer les métriques suivantes pour chaque modèle :
- Précision (accuracy)
- Temps d'entraînement
- Temps d'inférence (prédiction)

🚀 Modèles Utilisés
1. MobileNetV1
MobileNetV1 est un modèle de réseau de neurones convolutifs léger et rapide, adapté aux dispositifs mobiles. Il a été utilisé au cours du projet 5.

2. EfficientNetV2B0
EfficientNetV2B0 est une version améliorée de la famille des modèles EfficientNet, plus performant et efficace en termes de vitesse et de précision, tout en étant plus petit et rapide que son prédécesseur.

⚙️ Fonctionnalités du Dashboard
Le tableau de bord, réalisé avec Streamlit, permet à l'utilisateur de :
- Explorer le jeu de données : Visualiser des images et découvrir les différentes races de chiens présentes dans le jeu de données.
- Comparer les modèles : Afficher et comparer les métriques de performance (précision, temps d'entraînement et temps d'inférence) entre les modèles MobileNetV1 et EfficientNetV2B0.
- Uploader une image de chien : Tester le nouveau modèle EfficientNetV2B0 en téléchargeant une image de chien et obtenir une prédiction de la race.

📊 Comparaison des Modèles
1. Précision (Accuracy)
EfficientNetV2B0 montre une amélioration notable en termes de précision par rapport à MobileNetV1.

2. Temps d'entraînement
Le modèle MobileNetV1 est plus rapide à l'entraînement mais EfficientNetV2B0 reste dans les mêmes ordres de grandeur.

4. Temps d'inférence
MobileNetV1 a un temps d'inférence plus rapide qu'EfficientNetV2B0 mais les deux modèles restent dans les mêmes ordres de grandeur.

🖼️ Tester une Image
Le dashboard permet à l'utilisateur de télécharger une image d'un chien pour tester le modèle EfficientNetV2B0. Le modèle effectuera une prédiction et indiquera la race du chien.

📚 Technologies Utilisées
Python : Langage principal du projet
TensorFlow/Keras : Framework de deep learning utilisé pour entraîner et évaluer les modèles
EfficientNetV2B0 & MobileNetV1 : Modèles de classification d'images
Streamlit : Bibliothèque utilisée pour la création du tableau de bord interactif
Pandas & NumPy : Manipulation et analyse de données
Matplotlib & Seaborn : Visualisation des données et des résultats
