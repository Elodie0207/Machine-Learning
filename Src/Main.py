import os
from PIL import Image
import numpy as np
import shutil
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


def JeuDeDonnee(repertoire):
    for split in ['train', 'val', 'test']:
        for classe in ['voitures', 'motos']:
            os.makedirs(os.path.join('data', split, classe), exist_ok=True)

    for classe in ['voitures', 'motos']:
        images_classe = os.listdir(os.path.join(repertoire, classe))
        n = len(images_classe)

        train_size = int(0.7 * n)
        val_size = int(0.15 * n)
        test_size = n - train_size - val_size

        train_images = images_classe[:train_size]
        val_images = images_classe[train_size:train_size+val_size]
        test_images = images_classe[train_size+val_size:]


        for img in train_images:
            shutil.move(os.path.join(repertoire, classe, img), os.path.join('data', 'train', classe, img))
        for img in val_images:
            shutil.move(os.path.join(repertoire, classe, img), os.path.join('data', 'val', classe, img))
        for img in test_images:
            shutil.move(os.path.join(repertoire, classe, img), os.path.join('data', 'test', classe, img))


pretraiter_images(repertoire_voitures, output_repertoire + "\\voitures")
pretraiter_images(repertoire_motos, output_repertoire + "\\motos")
