from data_loader import setup_data_loader
from neural_trainer import train_model
from config import batch_size, learning_rate, epochs

def main():
    # Configura los DataLoader usando la configuración desde config.py
    train_loader, test_loader = setup_data_loader('mnist_data', batch_size)

    # Entrena el modelo con los parámetros especificados en config.py
    train_model(train_loader, epochs, learning_rate)

if __name__ == '__main__':
    main()