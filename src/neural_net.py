import torch.nn as nn

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(28 * 28, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )
    
    def forward(self, input_image):
        input_image = input_image.view(-1, 28 * 28)  # Conversión de imagen a vector
        output = self.model(input_image)  # Paso por todas las capas
        return output
