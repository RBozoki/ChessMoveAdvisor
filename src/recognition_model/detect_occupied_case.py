import os
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Paramètres
BOARD_IMAGES_DIR = "path_to_boards"  # Dossier contenant les images des plateaux
BOARD_FENS_FILE = "path_to_fens.txt"  # Fichier texte contenant les FEN correspondants
IMG_SIZE = (64, 64)  # Taille des cases après découpage
BATCH_SIZE = 32
EPOCHS = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# **Parsing et préparation des données**
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


def split_board_image(image, fen):
    """
    Découper une image de plateau en 64 sous-images et générer leurs labels d'occupation.
    """
    board = parse_fen(fen)  # Générer le tableau 8x8 des labels
    cell_size_x = image.shape[1] // 8
    cell_size_y = image.shape[0] // 8

    images = []
    labels = []

    for i in range(8):
        for j in range(8):
            # Découper chaque case
            x_start, x_end = j * cell_size_x, (j + 1) * cell_size_x
            y_start, y_end = i * cell_size_y, (i + 1) * cell_size_y
            cell = image[y_start:y_end, x_start:x_end]
            # Redimensionner la case
            cell = cv2.resize(cell, IMG_SIZE)
            images.append(cell)
            labels.append(board[i, j])  # 1 = occupé, 0 = vide

    return images, labels


class ChessDataset(Dataset):
    """
    Dataset PyTorch pour charger les cases d'échecs et leurs labels.
    """

    def __init__(self, board_images_dir, board_fens_file, transform=None):
        self.transform = transform
        self.images = []
        self.labels = []

        # Lire les FEN
        with open(board_fens_file, "r") as f:
            fens = f.readlines()

        # Associer chaque image à son FEN
        for idx, fen in enumerate(fens):
            board_image_path = os.path.join(board_images_dir, f"board_{idx}.png")
            if not os.path.exists(board_image_path):
                print(f"Image manquante : {board_image_path}")
                continue
            # Charger l'image et découper
            image = cv2.imread(board_image_path)
            board_images, board_labels = split_board_image(image, fen.strip())
            self.images.extend(board_images)
            self.labels.extend(board_labels)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        else:
            image = torch.tensor(image).permute(2, 0, 1).float() / 255.0  # Normalisation
        return image, torch.tensor(label).float()


# Charger le dataset
dataset = ChessDataset(BOARD_IMAGES_DIR, BOARD_FENS_FILE)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
train_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, test_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class ChessCNN(nn.Module):
    def __init__(self):
        super(ChessCNN, self).__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, activation='relu'),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc_layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 6 * 6, 128),  # Ajuster la taille si nécessaire
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),  # Une sortie pour la classification binaire
            nn.Sigmoid(),  # Activation pour les probabilités
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.fc_layers(x)
        return x


model = ChessCNN().to(DEVICE)
criterion = nn.BCELoss()  # Perte binaire croisée
optimizer = optim.Adam(model.parameters(), lr=0.001)


def train_model(model, train_loader, optimizer, criterion, epochs=10):
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Remise à zéro des gradients
            optimizer.zero_grad()

            # Passage avant et calcul de la perte
            outputs = model(images).squeeze()
            loss = criterion(outputs, labels)
            loss.backward()  # Backpropagation
            optimizer.step()  # Mise à jour des poids

            running_loss += loss.item()

        print(f"Époque {epoch + 1}/{epochs}, Perte : {running_loss / len(train_loader):.4f}")


train_model(model, train_loader, optimizer, criterion, epochs=EPOCHS)


def evaluate_model(model, test_loader):
    model.eval()
    y_true = []
    y_pred = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Prédictions
            outputs = model(images).squeeze()
            predictions = (outputs > 0.5).float()  # Seuil pour les probabilités

            y_true.extend(labels.cpu().numpy())
            y_pred.extend(predictions.cpu().numpy())

    # Rapport de classification
    print(classification_report(y_true, y_pred, target_names=["empty", "occupied"]))


evaluate_model(model, test_loader)
