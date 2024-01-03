import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

repertoire_voitures = "..\\Voitures"
repertoire_motos = "..\\Motos"
output_repertoire = "..\\Test"

class LinearModel:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def preprocess_images(self,repertoire_voitures, repertoire_motos, output_repertoire):
        for directory, output_dir in [(repertoire_voitures, output_repertoire + "\\voitures"),
                                      (repertoire_motos, output_repertoire + "\\motos")]:
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
        for directory in [output_repertoire + "\\voitures", output_repertoire + "\\motos"]:
            for filename in os.listdir(directory):
                img_path = os.path.join(directory, filename)
                img = Image.open(img_path)
                img_array = np.array(img) / 255.0
                if "voitures" in directory:
                    cars.append(img_array.flatten())
                else:
                    bikes.append(img_array.flatten())

        X = np.vstack((cars, bikes))
        y = np.hstack((np.zeros(len(cars)), np.ones(len(bikes))))

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
    model = LinearModel(learning_rate=0.001, n_iterations=1000)


    X, y = model.preprocess_images(repertoire_voitures, repertoire_motos, output_repertoire)


    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X = X[indices]
    y = y[indices]

    input_size = X.shape[1]

    model.fit(X, y)

    predictions = model.predict(X)

    class_names = {0: 'Voiture', 1: 'Motos'}
    predicted_classes = [class_names[pred] for pred in predictions]

    plt.figure(figsize=(8, 6))
    plt.scatter(range(len(predictions)), predictions, c=y, cmap='coolwarm', edgecolors='k')
    plt.colorbar(label='Classe (0 = Voiture, 1 = Moto)')
    plt.title('Prédictions du modèle linéaire')
    plt.xlabel('Échantillons')
    plt.ylabel('Prédictions')
    plt.show()

    plt.figure(figsize=(12, 8))
    for i in range(len(predictions)):
        plt.subplot(3, len(predictions)//3 + 1, i+1)
        plt.xticks([]), plt.yticks([])  # Hide tick marks
        plt.imshow(X[i].reshape(224, 224, 3))
        plt.xlabel(f'Prediction: {predicted_classes[i]}')
        plt.ylabel(f'Realite: {class_names[int(y[i])]}', rotation=0, labelpad=30)
    plt.suptitle('Modele lineaire avec image')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust subplot layout
    plt.show()
