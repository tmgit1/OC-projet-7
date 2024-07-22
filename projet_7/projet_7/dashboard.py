import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image, ImageOps
from utils.model_trained import model_prediction
import random
import os

# Configuration de la page Streamlit
st.set_page_config(page_title="Comparaison des Modèles de Classification de Chiens", layout="wide")

# Titre de la page
st.title("Analyse exploratoire des données du Stanford Dogs Dataset")

# Affichage de l'image avec un titre
st.markdown("## Exemples d'images du dataset")
image_path = "projet_7/projet_7/dashboard_eda_image_samples.png"
image = Image.open(image_path)
st.image(image, use_column_width=True)

# Spacing
st.markdown("---")

# Titre de la partie égalisation d'histogramme
st.title("Exemple d'égalisation d'histogramme")

# Liste de toutes les races dans le fichier "Images"
breed_dirs = [d for d in os.listdir("projet_7/projet_7/subset_images") if os.path.isdir(os.path.join("projet_7/projet_7/subset_images", d))]

# Dropdown menu pour sélectionner la race du chien
selected_breed = st.selectbox("Sélectionnez une race de chien", breed_dirs)

# Liste de toutes les images dans le fichier de la race sélectionnée
image_files = [f for f in os.listdir(os.path.join("projet_7/projet_7/subset_images", selected_breed)) if f.endswith(".jpg")]

# Sélection d'une image aléatoire dans la liste
random_image_file = random.choice(image_files)

image_path = os.path.join("projet_7/projet_7/subset_images", selected_breed, random_image_file)
image = Image.open(image_path)

# Affichage de l'image
st.image(image, width=300)

def histogram_equalization(img):
    # Convert the image to YUV color space
    img_yuv = img.convert('YCbCr')
    # Split the image into Y, Cb, and Cr channels
    y, cb, cr = img_yuv.split()
    # Apply histogram equalization to the Y channel
    y_equalized = ImageOps.equalize(y)
    # Merge the equalized Y channel with the original Cb and Cr channels
    img_equalized = Image.merge('YCbCr', (y_equalized, cb, cr))
    # Convert the image back to RGB color space
    img_output = img_equalized.convert('RGB')
    return img_output


# Bouton pour appliquer l'égalisation d'histogramme
if st.button("Appliquer l'égalisation d'histogramme"):
    equalized_image = histogram_equalization(image)
    st.image(equalized_image, width=300)

# DataFrame avec les résultats d'entraînement
data = {
    "Model": ["MobileNetV1", "MobileNetV1", "EfficientNetV2B0", "EfficientNetV2B0"],
    "Epochs": [20, 40, 20, 40],
    "Accuracy": [0.74, 0.75, 0.81, 0.82],
    "Training Time (hours)": [2.52, 5.43, 4.12, 8.56]
}

df = pd.DataFrame(data)

# Spacing
st.markdown("---")

# Titre de la page pour la partie comparaison des modèles
st.title("Comparaison des Modèles de Classification de Chiens")

# Ajout des descriptions textuelles pour les graphiques
st.markdown("Ce tableau de bord compare les performances des modèles MobileNetV1 et EfficientNetV2B0 en termes de précision (accuracy) et de temps d'entraînement pour 20 et 40 epochs.")

# Sélection des epochs
epochs = st.selectbox("Sélectionner le nombre d'epochs", options=[20, 40], help="Choisissez le nombre d'épochs pour voir les résultats correspondants.")

# Filtrage des données en fonction des epochs sélectionnées par l'utilisateur
filtered_df = df[df["Epochs"] == epochs]

# Création des graphiques
fig_accuracy = px.bar(
    filtered_df,
    x="Model",
    y="Accuracy",
    color="Model",
    title="Accuracy des modèles en fonction des epochs",
    labels={"Accuracy": "Accuracy"},
    text="Accuracy",  # Ajout des labels sur les barres pour plus de clarté,
    color_discrete_sequence=['#377eb8', '#ff7f00']  # Définir les couleurs manuellement
)

fig_accuracy.update_layout(
    coloraxis_showscale=False,  # Ne pas utiliser de barres de couleur
    yaxis=dict(tickformat=".0%"),  # Format des labels en pourcentage
    title=dict(font=dict(size=14)),  # Ajustement de la taille de la police du titre
    margin=dict(l=20, r=20, t=50, b=20)  # Réduction des marges pour plus d'espace
)

fig_time = px.bar(
    filtered_df,
    x="Model",
    y="Training Time (hours)",
    color="Model",
    title="Temps d'entraînement des modèles en fonction des epochs",
    labels={"Training Time (hours)": "Training Time (hours)"},
    text="Training Time (hours)",  # Ajout des labels sur les barres pour plus de clarté,
    color_discrete_sequence=['#377eb8', '#ff7f00']  # Définir les couleurs manuellement
)

fig_time.update_layout(
    coloraxis_showscale=False,  # Ne pas utiliser de barres de couleur
    title=dict(font=dict(size=14)),  # Ajustement de la taille de la police du titre
    margin=dict(l=20, r=20, t=50, b=20)  # Réduction des marges pour plus d'espace
)

# Affichage des graphiques côte à côte dans Streamlit avec `use_container_width=True`
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_accuracy, use_container_width=True)

with col2:
    st.plotly_chart(fig_time, use_container_width=True)

# Données pour le temps d'inférence
data_inference = {
    "Model": ["MobileNetV1", "EfficientNetV2B0"],
    "Total Inference Time (s)": [109, 120],  # en secondes
    "Inference Time per Image (s)": [0.0505, 0.0564]
}

df_inference = pd.DataFrame(data_inference)

# Menu déroulant
option = st.selectbox(
    "Choisissez l'affichage du temps d'inférence:",
    ("Temps total sur toutes les images", "Temps par image")
)

# Sélection des colonnes appropriées en fonction de l'option choisie par l'utilisateur
if option == "Temps total sur toutes les images":
    selected_column = "Total Inference Time (s)"
    ylabel = "Temps de prédiction total sur 2153 images (s)"
else:
    selected_column = "Inference Time per Image (s)"
    ylabel = "Temps de prédiction par image (s)"

# Création du graphique
fig_inference_time = px.bar(
    df_inference,
    x="Model",
    y=selected_column,
    color="Model",
    title="Temps d'inférence par modèle",
    labels={selected_column: ylabel},
    text=selected_column,  # Ajout des labels sur les barres pour plus de clarté
    color_discrete_sequence=['#377eb8', '#ff7f00']  # Définir les couleurs manuellement
)

fig_inference_time.update_layout(
    coloraxis_showscale=False,  # Ne pas utiliser de barres de couleur
    title=dict(font=dict(size=14)),  # Ajustement de la taille de la police du titre
    margin=dict(l=20, r=20, t=50, b=20)  # Réduction des marges pour plus d'espace
)

# Affichage du graphique dans Streamlit
st.plotly_chart(fig_inference_time, use_container_width=True)

# Alternative textuelle pour les graphiques
st.markdown("""
### Description des graphiques
- **Accuracy des modèles** : Ce graphique montre la précision des modèles MobileNetV1 et EfficientNetV2B0 pour 20 et 40 epochs. EfficientNetV2B0 présente une meilleure précision que MobileNetV1 pour les deux nombres d'epochs.
- **Temps d'entraînement** : Ce graphique indique le temps d'entraînement des modèles MobileNetV1 et EfficientNetV2B0 pour 20 et 40 epochs. MobileNetV1 a un temps d'entraînement plus court que EfficientNetV2B0.
- **Temps de prédiction** : Ce graphique indique le temps de prédiction des modèles MobileNetV1 et EfficientNetV2B0 sur le nombre total d'images de test, et le temps de prédiction par image. Les deux modèles ont un temps de prédiction du même ordre de grandeur avec un temps légèrement inférieur pour MobileNetV1.
""")

# Spacing
st.markdown("---")

# Titre pour la partie prédiction du modèle
st.markdown("## Chargez une image de chien")

upload = st.file_uploader("", type=['jpeg', 'jpg'])

c1, c2 = st.columns(2)

if upload:
    # Chargement de l'image
    image = Image.open(upload)

    # Pprediction
    predicted_class, probability = model_prediction(image)

    # Affichage de l'image
    c1.image(image)

    # Affichage de l'output du modèle
    c2.write("Race du chien:")
    c2.write(predicted_class)
    c2.write(f"Confiance: {probability:.2%}")
