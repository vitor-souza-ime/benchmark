import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets

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

# Transformação para o conjunto de teste (sem data augmentation)
test_transform = transforms.Compose([
    transforms.Resize((150, 150)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Carregar imagens da pasta dataset/train e dataset/test
train_dataset = ImageFolder('dataset/train', transform=transform)
test_dataset = ImageFolder('dataset/test', transform=test_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# Verificar se o modelo já existe. Se sim, carregar o modelo, senão, criar e treinar o modelo.
if os.path.exists(model_path):
    print("Carregando o modelo existente...")
    model = CNNModel().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()
else:
    print("Treinando o modelo...")
    start_time = time.time()
    model = CNNModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(EPOCA):
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
        
        accuracy = correct_predictions / total_samples
        print(f"Época {epoch + 1}, Loss: {running_loss / len(train_loader):.4f}, Acurácia: {accuracy:.4f}")
    
    torch.save(model.state_dict(), model_path)
    training_time = time.time() - start_time
    print(f"Modelo salvo em {model_path}")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")

# Fazer a predição e gerar a matriz de confusão
model.eval()  # Modo de avaliação
y_true = []
y_pred = []

with torch.no_grad():
    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

# Relatório de classificação
print("\nRelatório de classificação:")
target_names = list(test_dataset.class_to_idx.keys())
print(classification_report(y_true, y_pred, target_names=target_names))

# Plotar a matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred)
# Plotar a matriz de confusão com fonte maior
plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='d', 
            xticklabels=target_names, 
            yticklabels=target_names, 
            annot_kws={"size": 20})  # Define o tamanho da fonte das anotações
plt.xlabel('Classe Predita', fontsize=16)  # Define o tamanho da fonte do eixo X
plt.ylabel('Classe Verdadeira', fontsize=16)  # Define o tamanho da fonte do eixo Y
plt.title('Matriz de Confusão - PyTorch', fontsize=18)  # Define o tamanho da fonte do título
plt.xticks(fontsize=12)  # Define o tamanho da fonte dos rótulos do eixo X
plt.yticks(fontsize=12)  # Define o tamanho da fonte dos rótulos do eixo Y
plt.show()

