import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

repertoire_voitures = "..\\Voitures"
repertoire_motos = "..\\Motos"
output_repertoire = "..\\Test"

def pretraiter_images(repertoire, output_repertoire):
    for nom_fichier in os.listdir(repertoire):
        chemin_image = os.path.join(repertoire, nom_fichier)
        img = Image.open(chemin_image)


        img = img.resize((224, 224))


        img_array = np.array(img) / 255.0


        img_normalisee = Image.fromarray((img_array * 255).astype(np.uint8))


        chemin_sortie = os.path.join(output_repertoire, nom_fichier)
        img_normalisee.save(chemin_sortie)




def charger_images(repertoire):
    images = []
    for nom_fichier in os.listdir(repertoire):
        chemin_image = os.path.join(repertoire, nom_fichier)
        img = Image.open(chemin_image)
        img_array = np.array(img) / 255.0  # Normalisation
        images.append(img_array.flatten())  # Convertir l'image en un vecteur 1D et l'ajouter à la liste
    return images

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def entrainer_modele(X, y, epochs, learning_rate):
    # Initialiser les poids avec des valeurs aléatoires
    nombre_features = X.shape[1]
    poids = np.random.randn(nombre_features, 1)
    biais = 0

    # Entraînement du modèle
    for epoch in range(epochs):
        # Calculer les prédictions
        predictions = sigmoid(np.dot(X, poids) + biais)

        # Calculer l'erreur
        erreur = predictions - y.reshape(-1, 1)

        # Mettre à jour les poids et le biais en utilisant la descente de gradient
        gradient_poids = np.dot(X.T, erreur) / len(X)
        gradient_biais = np.sum(erreur) / len(X)

        poids -= learning_rate * gradient_poids
        biais -= learning_rate * gradient_biais

    return poids, biais

# Prétraitement des images
pretraiter_images(repertoire_voitures, output_repertoire + "\\voitures")
pretraiter_images(repertoire_motos, output_repertoire + "\\motos")

# Charger les images prétraitées
voitures = charger_images(output_repertoire + "\\voitures")
motos = charger_images(output_repertoire + "\\motos")

print("Forme des images de voitures :", [img.shape for img in voitures])
print("Forme des images de motos :", [img.shape for img in motos])
# Créer les étiquettes pour les données (0 pour les voitures, 1 pour les motos)
etiquettes_voitures = np.zeros(len(voitures))
etiquettes_motos = np.ones(len(motos))

# Concaténer les données d'entraînement
donnees_x = np.concatenate((voitures, motos))
donnees_y = np.concatenate((etiquettes_voitures, etiquettes_motos))

# Mélanger les données
indices = np.random.permutation(len(donnees_x))
donnees_x = donnees_x[indices]
donnees_y = donnees_y[indices]

# Entraîner le modèle
poids, biais = entrainer_modele(donnees_x, donnees_y, epochs=10000, learning_rate=0.001)


def projection_2d(images):
    # Prendre les deux premières dimensions des images aplaties
    return images[:, :2]

donnees_2d = projection_2d(donnees_x)

# Séparer les données en fonction des étiquettes
voitures_2d = donnees_2d[donnees_y == 0]
motos_2d = donnees_2d[donnees_y == 1]

# Afficher les données sur un graphique
plt.scatter(voitures_2d[:, 0], voitures_2d[:, 1], label='Voitures')
plt.scatter(motos_2d[:, 0], motos_2d[:, 1], label='Motos')
plt.title('Visualisation des données de voitures et motos')

# Dessiner la frontière de décision basée sur les poids et le biais du modèle
x_values = np.linspace(np.min(donnees_2d[:, 0]), np.max(donnees_2d[:, 0]), 100)
y_values = (-biais - poids[0] * x_values) / poids[1]
plt.plot(x_values, y_values, color='black', linestyle='--', label='Frontière de décision')

plt.xlabel('Dimension 1')
plt.ylabel('Dimension 2')
plt.legend()
plt.show()
