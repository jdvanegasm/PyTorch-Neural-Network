import torch.nn as nn

# Definicion de la red neuronal
class Classifier (nn.Module):
    def __init__(self):
        super(Classifier,self).__init__()
        self.input_layer = nn.Linear(28 * 28, 100)
        self.hidden_layer = nn.Linear(100, 50)
        self.output_layer = nn.Linear(50, 10)
        self.activation = nn.ReLU()
    def foward(self, input_image):
        input_image = input_image.view(-1, 28 * 28)
    # Conversión de imagen a vector
        output = self.activation(self.input_layer(input_image)) # Paso por la capa de entrada
        output = self.activation(self.hidden_layer(output)) # Paso por la capa oculta
        output = self.output_layer(output) # Paso por la capa de salida
        return output
