import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

repertoire_voitures = "..\\Voitures"
repertoire_motos = "..\\Motos"
repertoire_velos= "..\\Velos"
output_repertoire = "..\\Test"

np.random.seed(42)
X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)
class LinearModel:
    def __init__(self, learning_rate=0.00001, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def preprocess_images(self, repertoire_voitures, repertoire_motos, repertoire_velos, output_repertoire):
        for directory, output_dir in [(repertoire_voitures, output_repertoire + "\\voitures"),
                                      (repertoire_motos, output_repertoire + "\\motos"),
                                      (repertoire_velos, output_repertoire + "\\velos")]:  # Ajout du répertoire vélos
            for filename in os.listdir(directory):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path)
                img = img.resize((224, 224))
                img_array = np.array(img) / 255.0
                img_normalized = Image.fromarray((img_array * 255).astype(np.uint8))
                output_path = os.path.join(output_dir, filename)
                img_normalized.save(output_path)

        cars = []
        bikes = []
        velos=[]
        for directory in [output_repertoire + "\\voitures", output_repertoire + "\\motos", output_repertoire + "\\velos"]:
            for filename in os.listdir(directory):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path)
                img_array = np.array(img) / 255.0
                if "voitures" in directory:
                    cars.append(img_array.flatten())
                elif "velos" in directory:
                      velos.append(img_array.flatten())
                else:
                    bikes.append(img_array.flatten())

        X = np.vstack((cars, bikes, velos))
        y = np.hstack((np.zeros(len(cars)), np.ones(len(bikes)), np.full(len(velos), 2)))

        return X, y

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0

        for _ in range(self.n_iterations):
            predictions = np.dot(X, self.weights) + self.bias
            activations = np.where(predictions >= 0, 1, 0)

            dw = (1 / n_samples) * np.dot(X.T, (activations - y))
            db = (1 / n_samples) * np.sum(activations - y)

            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

    def predict(self, X):
        predictions = np.dot(X, self.weights) + self.bias
        return np.where(predictions >= 0, 1, 0)

if __name__ == "__main__":
    model = LinearModel(learning_rate=0.01, n_iterations=100000)
    X, y = model.preprocess_images(repertoire_voitures, repertoire_motos,repertoire_velos, output_repertoire)

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    model.fit(X, y)

    predictions = model.predict(X)

    class_names = {0: 'Voiture', 1: 'Motos', 2: 'Velos'}
    predicted_classes = [class_names[pred] for pred in predictions]
    plt.figure(figsize=(14, 8))
    plt.scatter(range(len(predictions)), predictions, c=y, cmap='coolwarm', edgecolors='k')
    plt.colorbar(label='Classe (0 = Voiture, 1 = Moto, 2 = Velos)')
    plt.title('Prédictions du modèle linéaire')
    plt.xlabel('Échantillons')
    plt.ylabel('Probabilité prédite')
    plt.show()

    n_images = len(predictions)
    n_rows = 5
    n_cols = (n_images + n_rows - 1) // n_rows

    plt.figure(figsize=(2 * n_cols, 2 * n_rows))
    for i in range(n_images):
        plt.subplot(n_rows, n_cols, i + 1)
        plt.xticks([]), plt.yticks([])  # Hide tick marks
        plt.imshow(X[i].reshape(224, 224, 3))
        plt.xlabel(f'Prédiction: {predicted_classes[i]}')
        plt.title(f'Réel: {class_names[int(y[i])]}')
    plt.suptitle('Modèle linéaire avec image', y=1.05)

    plt.subplots_adjust(wspace=0.4, hspace=0.4)

    plt.show()
