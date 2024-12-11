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

    # Gerar e plotar matriz de confusão
    # Obter predições
    predictions = []
    true_labels = []
    for i in range(len(train_generator)):
        x_batch, y_batch = train_generator[i]
        predictions.extend(np.argmax(model.predict(x_batch), axis=1))
        true_labels.extend(np.argmax(y_batch, axis=1))

    # Criar matriz de confusão
    cm = confusion_matrix(true_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=train_generator.class_indices.keys(), 
                yticklabels=train_generator.class_indices.keys())
    plt.xlabel('Classe Previsão')
    plt.ylabel('Classe Real')
    plt.title('Matriz de Confusão')
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

    # Verificar se a confiança é baixa
    if confidence < 0.5:
        print(f"Classe prevista: Desconhecida (Confiabilidade muito baixa: {confidence:.2f})")
    else:
        print(f'Classe prevista: {class_names[predicted_class]}')
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
