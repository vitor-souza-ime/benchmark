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

EPOCA = 60
model_path = 'model_py.pth'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Função para criar o modelo
class CNNModel(nn.Module):
    def __init__(self):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 37 * 37, 128)  # Ajuste a dimensão final do flatten
        self.fc2 = nn.Linear(128, 4)  # 4 classes: ball, post, line, robot

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(-1, 64 * 37 * 37)
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
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # Normalização similar ao ImageNet
])

# Carregar imagens da pasta dataset/train
train_dataset = ImageFolder('dataset/train', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Configurações de treinamento
model = CNNModel().to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Verificar se o modelo já existe
if os.path.exists(model_path):
    print("Carregando o modelo existente...")
    model.load_state_dict(torch.load(model_path))
else:
    print("Treinando o modelo...")
    start_time = time.time()  # Início da medição de tempo
    epochs = EPOCA
    all_labels = []
    all_preds = []

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
            
            # Cálculo de acurácia
            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            # Guardar as labels reais e as predições para a matriz de confusão
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(predicted.cpu().numpy())
        
        # Cálculo e exibição da acurácia
        accuracy = correct_predictions / total_samples
        print(f"Época {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Acurácia: {accuracy:.4f}")

    torch.save(model.state_dict(), model_path)
    training_time = time.time() - start_time  # Tempo de treinamento
    print(f"Modelo salvo em {model_path}")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")

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
    'dataset_noise/train/ball/lower_500061_bright_3.jpg',
    'dataset_noise/train/line/upper_101087_bright_3.jpg',
    'dataset_noise/train/robot/upper_512453_bright_3.jpg',
    'dataset_noise/train/post/upper_999151_bright_3.jpg',    
]

for img_path in img_paths:
    fazer_predicao(img_path)