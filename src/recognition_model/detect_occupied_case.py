import os
import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, ToTensor, Normalize, Resize

from src.recognition_model.detect_board import detect_board


def parse_fen(fen):
    """
    Parse un FEN pour générer un tableau 8x8 indiquant les cases occupées (1) ou vides (0).
    """
    rows = fen.split(" ")[0].split("/")  # Extraire uniquement la partie des positions
    board = []
    for row in rows:
        row_data = []
        for char in row:
            if char.isdigit():  # Les chiffres indiquent des cases vides consécutives
                row_data.extend([0] * int(char))
            else:  # Les caractères indiquent des pièces
                row_data.append(1)
        board.append(row_data)
    return np.array(board)


class ChessBoardDataset(Dataset):
    """
    Dataset PyTorch pour charger les cases découpées d'un plateau d'échecs et leurs labels.
    """

    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.file_pairs = [
            (os.path.join(data_dir, f), os.path.join(data_dir, f.replace(".png", ".json")))
            for f in os.listdir(data_dir) if f.endswith(".png")
        ]

    def __len__(self):
        return len(self.file_pairs) * 64  # Chaque plateau a 64 cases

    def __getitem__(self, idx):
        # Identifier l'image et le fichier FEN correspondant
        board_idx = idx // 64  # Plateau concerné
        cell_idx = idx % 64  # Case du plateau

        image_path, json_path = self.file_pairs[board_idx]

        # Charger l'image et le FEN
        image = cv2.imread(image_path)
        image = detect_board(image)
        with open(json_path, "r") as f:
            fen = json.load(f)["fen"]

        # Découper les cases
        cell_size_x = image.shape[1] // 8
        cell_size_y = image.shape[0] // 8
        row, col = divmod(cell_idx, 8)
        x_start, x_end = col * cell_size_x, (col + 1) * cell_size_x
        y_start, y_end = row * cell_size_y, (row + 1) * cell_size_y
        cell = image[y_start:y_end, x_start:x_end]

        # Redimensionner la case et appliquer des transformations
        if self.transform:
            cell = self.transform(cell)
        else:
            cell = torch.tensor(cell).permute(2, 0, 1).float() / 255.0

        # Générer le label à partir du FEN
        board = parse_fen(fen)
        label = board[row, col]  # 1 = occupé, 0 = vide

        return cell, torch.tensor(label, dtype=torch.float)


# Transformation des images
transform = Compose([
    ToTensor(),
    Resize((64, 64)),  # Redimensionner les cases à une taille standard
    Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalisation
])


import torch.nn as nn
import torch.optim as optim

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),  # Ajuste si la taille des cases diffère
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Sortie pour classification binaire
            nn.Sigmoid(),  # Activation pour les probabilités
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


# Initialiser le modèle

import torch.nn as nn
import torch.optim as optim

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),  # Ajuste si la taille des cases diffère
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Sortie pour classification binaire
            nn.Sigmoid(),  # Activation pour les probabilités
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x




def train_model(model, train_loader, optimizer, criterion, epochs=10):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Époque {epoch + 1}/{epochs}, Perte : {running_loss / len(train_loader):.4f}")




def evaluate_model(model, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images).squeeze()
            predictions = (outputs > 0.5).float()

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())



# Charger le dataset
data_dir = "path_to_dataset"  # Remplace par le chemin de ton dataset
dataset = ChessBoardDataset(data_dir, transform=transform)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# Initialiser le modèle
model = ChessCNN().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
criterion = nn.BCELoss()  # Binary Cross-Entropy
optimizer = optim.Adam(model.parameters(), lr=0.001)

train_model(model, train_loader, optimizer, criterion, epochs=10)

evaluate_model(model, test_loader)
