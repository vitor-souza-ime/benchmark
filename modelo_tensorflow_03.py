import os
import time
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
import numpy as np

EPOCA=30

# Caminho para salvar o modelo
model_path = 'model_tf.h5'

# Função para criar o modelo
def criar_modelo():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(4, activation='softmax')  # 4 classes: ball, post, line, robot
    ])
    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Gerador de dados com data augmentation
datagen = ImageDataGenerator(
    rescale=1./255,  # Normalizar os valores dos pixels para [0, 1]
    brightness_range=[0.2, 1.0],  # Variação do brilho
    zoom_range=0.2,  # Aplicar zoom
    horizontal_flip=True,  # Inversão horizontal
    rotation_range=30,  # Rotacionar as imagens em até 30 graus
    width_shift_range=0.2,  # Deslocamento horizontal
    height_shift_range=0.2,  # Deslocamento vertical
    shear_range=0.2,  # Aplicar distorção (shear)
    channel_shift_range=100.0,  # Mudança nos canais de cor (alteração de saturação/cor)
    fill_mode='nearest'  # Preenchimento de áreas vazias resultantes da transformação
)

# Carregar imagens da pasta dataset/train
train_generator = datagen.flow_from_directory(
    'dataset/train',  # Caminho do diretório de treino
    target_size=(150, 150),  # Redimensionar as imagens para 150x150 pixels
    batch_size=32,  # Tamanho do lote (batch size)
    class_mode='categorical'  # Modo de classificação: várias classes (categorical)
)

# Verificar se o modelo já existe. Se sim, carregar o modelo, senão, criar e treinar o modelo.
if os.path.exists(model_path):
    print("Carregando o modelo existente...")
    model = load_model(model_path)
    model.summary()
else:
    print("Treinando o modelo...")
    start_time = time.time()  # Início da medição de tempo
    model = criar_modelo()
    model.fit(train_generator, epochs=EPOCA)
    model.save(model_path)
    training_time = time.time() - start_time  # Tempo de treinamento
    print(f"Modelo salvo em {model_path}")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")

# Função para fazer a predição em uma imagem específica
def fazer_predicao(img_path):
    print()
    # Carregar a imagem e redimensionar
    img = load_img(img_path, target_size=(150, 150))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Adicionar dimensão para batch
    img_array /= 255.0  # Normalizar

    # Fazer a predição e medir o tempo
    start_time = time.time()  # Início da medição de tempo
    prediction = model.predict(img_array)
    prediction_time = time.time() - start_time  # Tempo de predição

    # Obter a classe prevista e a confiabilidade
    predicted_class = np.argmax(prediction)
    confidence = np.max(prediction)  # Confiança da predição

    # Mapear índices de classes para nomes das classes
    class_indices = train_generator.class_indices  # Mapeamento das classes
    class_names = list(class_indices.keys())  # Obter nomes das classes

    
    # Verificar se a confiança é baixa
    if confidence < 0.5:
        print(f"Classe prevista: Desconhecida (Confiabilidade muito baixa: {confidence:.2f})")
    else:
        print(f'Classe prevista: {class_names[predicted_class]}')
        print(f'Confiabilidade: {confidence:.2f}')
    print(f"Tempo de predição: {prediction_time:.4f} segundos")

# Fazer a predição com a imagem 'dataset/train/ball/lower_496163.jpg'
img_path = 'dataset/train/line/upper_101224.jpg'
fazer_predicao(img_path)

img_path = 'dataset/train/robot/lower_936361.jpg'
fazer_predicao(img_path)

img_path = 'dataset/train/ball/lower_500228.jpg'
fazer_predicao(img_path)

img_path = 'dataset_noise/train/ball/lower_496163_bright_4.jpg'
fazer_predicao(img_path)

img_path = 'dataset_noise/train/ball/lower_496163_bright_2.jpg'
fazer_predicao(img_path)
