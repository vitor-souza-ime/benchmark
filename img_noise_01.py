import os
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image, ImageEnhance
import numpy as np

# Caminhos das pastas
dataset_dir = 'dataset'  # Pasta original do dataset
output_dir = 'dataset_noise'  # Pasta onde as imagens aumentadas serão salvas

# Se a pasta de saída não existir, criar
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Fatores de brilho para simular ambientes claros e escuros
brightness_factors = [0.35, 0.5, 1.0, 1.5, 2.0]  # Fatores para escurecer e clarear a imagem

# Função para ajustar o brilho da imagem
def adjust_brightness(img, factor):
    enhancer = ImageEnhance.Brightness(img)
    return enhancer.enhance(factor)

# Função para processar e salvar as imagens aumentadas
def process_and_save_images():
    # Percorrer todas as subpastas (classes) no dataset original
    for subdir, dirs, files in os.walk(dataset_dir):
        for file in files:
            # Verificar se o arquivo é uma imagem (com extensão comum)
            if file.endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(subdir, file)
                print(f"Processando imagem: {file}")
                img = load_img(img_path)  # Carregar imagem como objeto PIL
                prefix = os.path.splitext(file)[0]  # Nome base do arquivo

                # Gerar variações de brilho
                for i, brightness_factor in enumerate(brightness_factors):
                    img_bright = adjust_brightness(img, brightness_factor)

                    # Criar o caminho da pasta de saída (mesma estrutura de classes)
                    output_subdir = subdir.replace(dataset_dir, output_dir)
                    if not os.path.exists(output_subdir):
                        os.makedirs(output_subdir)
                    
                    # Salvar a imagem aumentada
                    output_path = os.path.join(output_subdir, f"{prefix}_bright_{i+1}.jpg")
                    img_bright.save(output_path)

# Processar as imagens
process_and_save_images()
