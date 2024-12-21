import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

EPOCA = 60

# Caminho para salvar o modelo
model_path = 'model_tf.h5'

# Função para criar o modelo
def criar_modelo():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3), padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu', padding='same'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')  # 4 classes: ball, post, line, robot
    ])
    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Gerador de dados com data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    brightness_range=[0.2, 1.0],
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    channel_shift_range=100.0,
    fill_mode='nearest'
)

# Carregar imagens da pasta dataset/train
train_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

# Verificar se o modelo já existe
if os.path.exists(model_path):
    print("Carregando o modelo existente...")
    model = load_model(model_path)
    model.summary()
else:
    print("Treinando o modelo...")
    start_time = time.time()
    model = criar_modelo()
    history = model.fit(train_generator, epochs=EPOCA)
    model.save(model_path)
    training_time = time.time() - start_time
    print(f"Modelo salvo em {model_path}")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")

    # Cálculo das médias quadráticas
    accuracies = np.array(history.history['accuracy'])
    losses = np.array(history.history['loss'])
    mean_squared_accuracy = np.sqrt(np.mean(accuracies**2))
    mean_squared_loss = np.sqrt(np.mean(losses**2))

    print("\nResultados Finais do Treinamento:")
    print(f"Média Quadrática da Acurácia: {mean_squared_accuracy:.4f}")
    print(f"Média Quadrática da Perda (Loss): {mean_squared_loss:.4f}")

    # Plotar gráficos de perda e acurácia por época
    plt.figure(figsize=(12, 6))

    # Gráfico de perda
    #plt.subplot(1, 2, 1)
    plt.plot(range(1, EPOCA + 1), losses, marker='o', linestyle='-', color='r', label='Loss')
    plt.xlabel('Época')
    plt.ylabel('Loss')
    plt.title('Loss por Época')
    plt.grid(True)
    plt.xticks(range(0, EPOCA + 1, 10))  # Eixo X espaçado de 10 em 10
    plt.legend()

    # Gráfico de acurácia
    #plt.subplot(1, 2, 2)
    plt.plot(range(1, EPOCA + 1), accuracies, marker='o', linestyle='-', color='b', label='Accuracy')
    plt.xlabel('Época')
    plt.ylabel('Acurácia')
    plt.title('Acurácia por Época')
    plt.grid(True)
    plt.xticks(range(0, EPOCA + 1, 10))  # Eixo X espaçado de 10 em 10
    plt.legend()

    # Exibir os gráficos
    plt.tight_layout()
    plt.show()

# Função para fazer a predição em uma imagem específica
def fazer_predicao(img_path):
    print()
    # Carregar a imagem e redimensionar
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Fazer a predição e medir o tempo
    start_time = time.time()
    prediction = model.predict(img_array)
    prediction_time = time.time() - start_time

    # Obter a classe prevista e a confiabilidade
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)

    # Mapear índices de classes para nomes das classes
    class_indices = train_generator.class_indices
    class_names = list(class_indices.keys())

    # Exibir a imagem junto com a previsão
    plt.imshow(load_img(img_path))
    plt.title(f'Classe Prevista: {class_names[predicted_class]}\nConfiabilidade: {confidence:.2f}')
    plt.axis('off')
    plt.show()

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
