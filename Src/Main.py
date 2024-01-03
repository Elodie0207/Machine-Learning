import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

class LinearRegression:
    def __init__(self, learning_rate=0.01, n_iterations=1000):
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.weights = None
        self.bias = None

    def initialize_weights(self, input_size):
        self.weights = np.random.randn(input_size + 1, 1)
        self.bias = 0

    def preprocess_image(self, img_path):
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_array = np.array(img) / 255.0
        return img_array.flatten()

    def load_images_from_directory(self, directory):
        images = []
        for filename in os.listdir(directory):
            img_path = os.path.join(directory, filename)
            img_array = self.preprocess_image(img_path)
            images.append(img_array)
        return images

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def loss(self, y_true, y_pred):
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.initialize_weights(n_features)
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        y = y.reshape(-1, 1)

        for _ in range(self.n_iterations):
            y_predicted = self.sigmoid(np.dot(X, self.weights) + self.bias)
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            #db = (1 / n_samples) * np.sum(y_predicted - y)

            self.weights -= self.learning_rate * dw
            #self.bias -= self.learning_rate * db

    def predict(self, X):
        X = np.hstack((np.ones((X.shape[0], 1)), X))
        #return self.sigmoid(np.dot(X, self.weights) + self.bias)
        return self.sigmoid(np.dot(X, self.weights))

if __name__ == "__main__":
    cars_directory = "../Voitures"
    bikes_directory = "../Motos"

    model = LinearRegression(learning_rate=0.01, n_iterations=1000)

    cars = model.load_images_from_directory(cars_directory)
    bikes = model.load_images_from_directory(bikes_directory)

    X = np.vstack((cars, bikes))
    y = np.hstack((np.zeros(len(cars)), np.ones(len(bikes))))

    X = np.hstack((np.ones((X.shape[0], 1)), X))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    predicted_classes = (predictions > 0.5).astype(int)

    accuracy = accuracy_score(y_test, predicted_classes)
    print(f"Accuracy: {accuracy}")

    conf_matrix = confusion_matrix(y_test, predicted_classes)
    print(f"Confusion Matrix:\n{conf_matrix}")

    plt.figure(figsize=(10, 6))
    plt.scatter(range(len(predictions)), predictions.flatten(), c=y_test.flatten(), cmap='coolwarm', edgecolors='k')
    plt.colorbar(label='Classe (0 = Voiture, 1 = Moto)')
    plt.title('Prédictions du modèle de régression linéaire')
    plt.xlabel('Échantillons')
    plt.ylabel('Probabilité prédite')
    plt.show()
