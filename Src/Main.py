import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

repertoire_voitures = "..\\Voitures"
repertoire_motos = "..\\Motos"
output_repertoire = "..\\Test"


class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def initialize_weights(self, input_size):
        self.weights = np.random.randn(input_size, 1)  # Initialisation aléatoire
        self.bias = 0
    def pretraiter_images(self, repertoire, output_repertoire):
        for nom_fichier in os.listdir(repertoire):
            chemin_image = os.path.join(repertoire, nom_fichier)
            img = Image.open(chemin_image)

            img = img.resize((224, 224))

            img_array = np.array(img) / 255.0

            img_normalisee = Image.fromarray((img_array * 255).astype(np.uint8))

            chemin_sortie = os.path.join(output_repertoire, nom_fichier)
            img_normalisee.save(chemin_sortie)

    def charger_images_repertoire(self, repertoire):
        images = []
        for nom_fichier in os.listdir(repertoire):
            chemin_image = os.path.join(repertoire, nom_fichier)
            img = Image.open(chemin_image)
            img_array = np.array(img) / 255.0
            images.append(img_array.flatten())
        return images

    def afficher_images(self, images, num_images_a_afficher=5, titre="Images"):
        fig, axes = plt.subplots(1, num_images_a_afficher, figsize=(15, 3))
        for i in range(min(num_images_a_afficher, len(images))):
            axes[i].imshow(images[i].reshape(224, 224, 3))  # Remettre l'image en forme pour l'affichage
            axes[i].axis('off')
            axes[i].set_title(f'{titre} {i + 1}')
        plt.tight_layout()
        plt.show()

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros((n_features, 1))  # Adapter les dimensions des poids
        self.bias = 0

        # Gradient descent
        for _ in range(self.n_iterations):
            y_predicted = np.dot(X, self.weights) + self.bias
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))

            # Ajuster les dimensions du gradient pour correspondre à self.weights
            dw = dw.reshape(self.weights.shape)

            db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias

# Exemple d'utilisation
if __name__ == "__main__":
    # Génération de nouvelles données de voitures et motos
    model = LinearRegression(learning_rate=0.01, n_iterations=1000)

    # Prétraitement des images
    model.pretraiter_images(repertoire_voitures, output_repertoire + "\\voitures")
    model.pretraiter_images(repertoire_motos, output_repertoire + "\\motos")

    # Charger les images prétraitées
    voitures = model.charger_images_repertoire(output_repertoire + "\\voitures")
    motos = model.charger_images_repertoire(output_repertoire + "\\motos")

    # Fusionner les données et les étiquettes
    X = np.vstack((voitures, motos))
    y = np.hstack((np.zeros(len(voitures)), np.ones(len(motos))))

    input_size = X.shape[1]  # Taille des données en entrée

    model.initialize_weights(input_size)

    # Entraîner le modèle de régression linéaire avec les images
    model.fit(X, y)

    # Prédiction sur les données d'entraînement
    predictions = model.predict(X)

    # Tracer le graphique scatter des prédictions
    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(predictions)), predictions, c=y, cmap='coolwarm', edgecolors='k')
    plt.colorbar(label='Classe (0 = Voiture, 1 = Moto)')
    plt.title('Prédictions du modèle de régression linéaire')
    plt.xlabel('Échantillons')
    plt.ylabel('Prédictions')
    plt.show()
