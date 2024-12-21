import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from torchsummary import summary

EPOCA = 60
model_path = 'model_py.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para criar o modelo
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 18 * 18, 128)
        self.fc2 = nn.Linear(128, 4)  # 4 classes: ball, post, line, robot

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.pool(torch.relu(self.conv3(x)))

        x = torch.flatten(x, 1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Transformações de data augmentation
transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carregar imagens da pasta dataset/train
train_dataset = ImageFolder('dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Configurações de treinamento
model = CNNModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# O restante do código permanece igual ao original
if os.path.exists(model_path):    
    print("Carregando o modelo existente...")
    model.load_state_dict(torch.load(model_path))
    print("\nSumário do Modelo:")
    summary(model, (3, 150, 150))    
else:
    print("Treinando o modelo...")
    start_time = time.time()
    epochs = EPOCA
    all_labels = []
    all_preds = []
    accuracies = []  # Lista para armazenar a acurácia de cada época
    losses = []  # Lista para armazenar a perda de cada época

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        
        accuracy = correct_predictions / total_samples
        accuracies.append(accuracy)
        losses.append(running_loss / len(train_loader))  # Armazenando a perda média
        print(f"Época {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Acurácia: {accuracy:.4f}")

    torch.save(model.state_dict(), model_path)
    training_time = time.time() - start_time
    print(f"Modelo salvo em {model_path}")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")

    # Calcular a média quadrática da acurácia
    mean_squared_accuracy = np.sqrt(np.mean(np.array(accuracies) ** 2))
    print(f"Média quadrática da acurácia: {mean_squared_accuracy:.4f}")

    # Calcular a média quadrática da perda (loss)
    mean_squared_loss = np.sqrt(np.mean(np.array(losses) ** 2))
    print(f"Média quadrática da perda: {mean_squared_loss:.4f}")

    # Plotar gráfico de acurácia por época
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), accuracies, marker='o', linestyle='-', color='b')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.title('Acurácia por Época')
    plt.grid(True)
    plt.xticks(ticks=range(0, epochs + 1, 10))  # Eixo X espaçado de 10 em 10
    plt.yticks(np.arange(0, 1.1, 0.5))  # Eixo Y espaçado de 0.5 em 0.5
    plt.show()

    # Plotar gráfico de perda por época
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), losses, marker='o', linestyle='-', color='r')
    plt.xlabel('Época')
    plt.ylabel('Perda')
    plt.title('Perda (Loss) por Época')
    plt.grid(True)
    plt.xticks(ticks=range(0, epochs + 1, 10))  # Eixo X espaçado de 10 em 10
    plt.yticks(np.arange(0, max(losses) + 0.1, 0.5))  # Eixo Y espaçado de 0.5 em 0.5
    plt.show()

    # Gerar a matriz de confusão
    cm = confusion_matrix(all_labels, all_preds)

    # Plotar a matriz de confusão
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=train_dataset.classes, yticklabels=train_dataset.classes)
    plt.xlabel('Classe Previsão')
    plt.ylabel('Classe Real')
    plt.title('Matriz de Confusão')
    plt.show()

# Função para fazer a predição em uma imagem específica
def fazer_predicao(img_path):
    img = Image.open(img_path)
    img = transform(img).unsqueeze(0).to(device)
    
    print()
    
    model.eval()
    start_time = time.time()
    with torch.no_grad():
        outputs = model(img)
        prediction_time = time.time() - start_time

    _, predicted_class = torch.max(outputs, 1)
    confidence = torch.softmax(outputs, dim=1)[0][predicted_class].item()

    class_indices = train_dataset.class_to_idx
    class_names = {v: k for k, v in class_indices.items()}
    
    if confidence < 0.5:
        print(f"Classe prevista: Desconhecida (Confiabilidade muito baixa: {confidence:.2f})")
    else:
        print(f'Classe prevista: {class_names[predicted_class.item()]}')
        print(f'Confiabilidade: {confidence:.2f}')
    print(f"Tempo de predição: {prediction_time:.4f} segundos")

# Fazer a predição com algumas imagens
img_paths = [
    'dataset_noise/train/ball/lower_500061_bright_4.jpg',
    'dataset_noise/train/ball/lower_500061_bright_2.jpg',
    'dataset_noise/train/line/upper_101087_bright_4.jpg',
    'dataset_noise/train/line/upper_101087_bright_2.jpg',
    'dataset_noise/train/robot/upper_512453_bright_4.jpg',
    'dataset_noise/train/robot/upper_512453_bright_2.jpg',
    'dataset_noise/train/post/upper_999151_bright_4.jpg',
    'dataset_noise/train/post/upper_999151_bright_2.jpg' 
]

for img_path in img_paths:
    fazer_predicao(img_path)
