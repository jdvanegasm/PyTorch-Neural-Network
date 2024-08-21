from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np
import torch as torch

def plot_losses(losses):
    plt.style.use("Solarize_Light2")
    plt.figure(figsize=(10, 5))
    plt.plot(np.arange(len(losses)), losses, label="Training Loss")
    plt.title("Loss During Training")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

def print_model_summary(model):
    summary(model, input_size=(1, 28, 28))  # Asumiendo que la entrada es 1x28x28 para MNIST

def sanity_check(model, device='cpu'):
    model.eval()
    with torch.no_grad():
        sample_input = torch.rand((1, 28 * 28)).to(device)
        output = model(sample_input)
        print("Output shape:", output.shape)  # Debería ser [1, 10] para MNIST
