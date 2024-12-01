import os
import json

import numpy as np
import torch
from PIL import Image
from matplotlib import pyplot as plt
from torchvision.transforms import ToTensor, Compose, Resize
from torchvision.transforms.functional import to_pil_image

# Transformation par défaut pour convertir les images en tenseurs
default_transform = Compose([Resize((224, 224)), ToTensor()])

def load_image(image_path, transform=None):
    """
    Charge une image PNG à partir du chemin donné et applique une transformation.
    """
    image = Image.open(image_path).convert("RGB")  # Convertit en RGB
    if transform:
        image = transform(image)
    return image

def load_fen(json_path):
    """
    Charge les données FEN depuis un fichier JSON.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data.get('fen', None)  # Récupère la clé "fen" ou None si absente

def load_dataset(folder_path, transform=None):
    """
    Charge les paires (image, FEN) depuis un dossier contenant des fichiers PNG et JSON.
    Retourne une liste de tuples (image, fen).
    """
    dataset = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".png"):
            base_name = os.path.splitext(filename)[0]
            image_path = os.path.join(folder_path, f"{base_name}.png")
            json_path = os.path.join(folder_path, f"{base_name}.json")

            if not os.path.exists(json_path):
                print(f"JSON file missing for {filename}, skipping...")
                continue  # Ignore si le JSON n'existe pas

            # Charger l'image et le FEN
            image = load_image(image_path, transform)
            fen = load_fen(json_path)
            dataset.append((image, fen))
    return dataset

def show_image(image, fen=None, title="Image", figsize=(6, 6)):
    if isinstance(image, torch.Tensor):
        # Convertir le tenseur en une image PIL
        image = to_pil_image(image)
    elif isinstance(image, np.ndarray):
        # Convertir le tableau NumPy en image
        if image.ndim == 2:  # Grayscale
            cmap = "gray"
        else:
            cmap = None
        plt.imshow(image, cmap=cmap)
    else:
        raise ValueError("Image must be a PyTorch tensor or a NumPy array.")

    # Affichage avec Matplotlib
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis("off")
    if fen:
        plt.title(f"FEN: {fen}")
    else:
        plt.title(title)
    plt.show()