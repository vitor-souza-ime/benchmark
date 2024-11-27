import os
import time
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import confusion_matrix, classification_report

EPOCA = 60

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
    rescale=1./255,
    brightness_range=[0.2, 1.0],
    zoom_range=0.2,
    horizontal_flip=True,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    channel_shift_range=100.0,
    fill_mode='nearest'
)

# Gerador de dados para o conjunto de teste (sem data augmentation)
test_datagen = ImageDataGenerator(rescale=1./255)

# Carregar imagens da pasta dataset/train e dataset/test
train_generator = datagen.flow_from_directory(
    'dataset/train',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical'
)

test_generator = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(150, 150),
    batch_size=32,
    class_mode='categorical',
    shuffle=False  # Importante para matriz de confusão
)

# Verificar se o modelo já existe. Se sim, carregar o modelo, senão, criar e treinar o modelo.
if os.path.exists(model_path):
    print("Carregando o modelo existente...")
    model = load_model(model_path)
    model.summary()
else:
    print("Treinando o modelo...")
    start_time = time.time()
    model = criar_modelo()
    model.fit(train_generator, epochs=EPOCA)
    model.save(model_path)
    training_time = time.time() - start_time
    print(f"Modelo salvo em {model_path}")
    print(f"Tempo de treinamento: {training_time:.2f} segundos")

# Fazer a predição e gerar a matriz de confusão
Y_pred = model.predict(test_generator)
y_pred_classes = np.argmax(Y_pred, axis=1)
y_true = test_generator.classes

# Relatório de classificação
print("\nRelatório de classificação:")
print(classification_report(y_true, y_pred_classes, target_names=list(test_generator.class_indices.keys())))

# Plotar a matriz de confusão
conf_matrix = confusion_matrix(y_true, y_pred_classes)
# Plotar a matriz de confusão com tamanho da letra ajustado
plt.figure(figsize=(10, 8))
sns.heatmap(
    conf_matrix,
    annot=True,
    cmap='Blues',
    fmt='d',
    xticklabels=test_generator.class_indices.keys(),
    yticklabels=test_generator.class_indices.keys(),
    annot_kws={"size": 14}  # Aumenta o tamanho do texto dentro dos quadrados
)
plt.xlabel('Classe Predita', fontsize=16)
plt.ylabel('Classe Verdadeira', fontsize=16)
plt.title('Matriz de Confusão - TensorFlow', fontsize=18)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.show()

