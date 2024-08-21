import torch
import torch.nn as nn
import torch.optim as optim
from neural_net import Classifier
from utils import plot_losses  # Importar la función para graficar las pérdidas

def train_model(data_loader, epochs, learning_rate):
    classifier = Classifier()
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(classifier.parameters(), lr=learning_rate)
    losses = []  # Lista para almacenar las pérdidas

    for e in range(epochs):
        for images, labels in data_loader:
            output = classifier(images)
            classifier.zero_grad()
            error = loss_function(output, labels)
            error.backward()
            optimizer.step()
            losses.append(error.item())

    plot_losses(losses)  # Llamar a la función para graficar las pérdidas al final del entrenamiento