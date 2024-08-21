import torchvision
import torch

T = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
images = torchvision.datasets.MNIST('mnist_data', transform = T, download = True)
image_loader = torch.util.data.DataLoader(images, batch_size = 128)