from torch.autograd import Variable
from neural_net import Classifier
from main import image_loader
from main import images
from torch import optim
import matplotlib.pyplot as plt
import torch.nn as nn
import numpy as np

classifier = Classifier() # Instanciación de la neural net
loss_function = nn.CrossEntropyLoss() # Función de Pérdidas
parameters = classifier.parameters()
optimizer = optim.Adam(params = parameters, lr = 0.001) # Optimización de los parametros
epochs = 3 # Numero de veces que pasamos cad amuestra a la Neural Net del Training
iterations = 0 # Numero total de iteraciones para mostrar error
losses = np.array([]) # Array que guarda la perdida en cada iteración

for e in range(epochs):
    for i, (images, tags) in enumerate(image_loader):
        image, tags = Variable(images), Variable(tags) # Conversion a variable para la derivacion
        output = Classifier(images) # Calcular la salida para una imagen
        Classifier.zero_grad() # Poner los gradientes a cero en cada iteración
        error = loss_function(output, tags) # Calcular el error
        error.backward() # Obtener los gradientes y propagar
        optimizer.step() # Actualizar los pesos con los gradientes
        iterations += 1
        losses = np.append(losses, error.item()) 
plt.style.use("seaborn-whitegrid") # Perdidas en cada iteracion de forma grafica
plt.plot(np.arange(iterations), losses)