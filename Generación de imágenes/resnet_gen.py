 
# ### Librerias


import tqdm


from urllib.request import urlopen
from PIL import Image
import timm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import copy
import EDAspy2
from EDAspy2.optimization import EBNA
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


import torch
import sys

# Comprobar la versión de Python
print(f"Versión de Python: {sys.version}")

# Comprobar la versión de PyTorch
print(f"Versión de PyTorch: {torch.__version__}")

# Comprobar la versión de CUDA disponible para PyTorch
print(f"Versión de CUDA disponible: {torch.version.cuda}")


print(torch.__version__)
print('CUDA disponible:', torch.cuda.is_available())
print('Nombre del dispositivo CUDA:', torch.cuda.get_device_name(0))

 
# ## Carga del modelo


from transformers import AutoImageProcessor, ResNetForImageClassification

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")

 
# ![image.png](attachment:image.png)




model = model.eval()  # Poner el modelo en modo de evaluación
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = model.to(device)


data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

 
# ### 1.1 Carga de datos


root_data="/home/v839/v839190/poda/poda efficientnet/data/"



# Define las rutas a los archivos
loc_synset_mapping_path = root_data+'LOC_synset_mapping.txt'
ids_path = root_data+'ids.txt'

# Inicializa un diccionario para el mapeo de Synset ID a Etiqueta Numérica
synset_to_num = {}
# Inicializa un diccionario para el mapeo de Etiqueta Numérica a Descripción Humana
num_to_human = {}

# Leer LOC_synset_mapping.txt y construir ambos mapeos
with open(loc_synset_mapping_path, 'r') as f:
    for index, line in enumerate(f):
        parts = line.strip().split(' ', 1)
        synset_id = parts[0]
        human_readable = parts[1] if len(parts) > 1 else ""

        # Asignar el índice como etiqueta numérica al synset ID
        synset_to_num[synset_id] = index
        # Asignar la descripción humana a la etiqueta numérica
        num_to_human[index] = human_readable

# Opcionalmente, imprime los primeros elementos de cada mapeo para verificar
print("Synset to Numeric Label Mapping (sample):", list(synset_to_num.items())[:5])
print("Numeric Label to Human-readable Mapping (sample):", list(num_to_human.items())[:5])

# Si necesitas el mapeo inverso de etiquetas numéricas a Synset IDs (por ejemplo, para usar con ids.txt),
# puedes invertir el diccionario synset_to_num así:
num_to_synset = {v: k for k, v in synset_to_num.items()}


from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, dataset, synset_to_num):
        self.dataset = dataset
        self.synset_to_num = synset_to_num
        # Invertir el mapeo de clase a índice de ImageFolder para obtener Synset IDs a partir de etiquetas de ImageFolder
        self.idx_to_synset = {v: k for k, v in self.dataset.class_to_idx.items()}

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        # Convertir la etiqueta de ImageFolder a Synset ID y luego a tu etiqueta numérica personalizada
        synset_id = self.idx_to_synset[label]
        custom_label = self.synset_to_num[synset_id]
        return img, custom_label








from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

data_dir = root_data+'imagenes'

# Transformaciones (asumiendo que ya las has definido)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Cargar todas las imágenes con ImageFolder
all_data = datasets.ImageFolder(root=data_dir, transform=transform)

# Envolver el dataset de ImageFolder en tu CustomDataset
custom_all_data = CustomDataset(all_data, synset_to_num)

# Dividir los datos en conjuntos de entrenamiento, validación y prueba
train_size = int(0.6 * len(custom_all_data))  # 60% de los datos para entrenamiento
valid_size = int(0.2 * len(custom_all_data))  # 20% de los datos para validación
test_size = len(custom_all_data) - train_size - valid_size  # El resto para prueba

train_dataset, valid_dataset, test_dataset = random_split(custom_all_data, [train_size, valid_size, test_size])

# DataLoader
batch_size = 32
data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)












import torch
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

# Función para obtener los logits y etiquetas en formato legible
def get_predictions_and_labels(data_loader, model):
    model.eval()  # Asegurarse de que el modelo está en modo evaluación
    with torch.no_grad():  # Desactivar el cálculo de gradientes
        for images, labels in data_loader:
            images = images.to(model.device)
            labels = labels.to(model.device)
            outputs = model(images)
            logits = outputs.logits
            return logits, labels, images  # Usa return aquí para solo procesar un batch


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch

# Función para obtener los logits y las etiquetas asociadas en un DataFrame
def logits_to_dataframe(logits, num_to_human):
    logits_array = logits.cpu().numpy()  # Convertir los logits a un array de numpy
    # Crear un DataFrame con los logits y las etiquetas humanas correspondientes
    data = {
        'Logit': logits_array,
        'Label': [num_to_human[i] for i in range(len(logits_array))]
    }
    df = pd.DataFrame(data)
    df.sort_values(by='Logit', ascending=False, inplace=True)  # Ordenar por logits, de mayor a menor
    return df

# Función para visualizar una imagen junto con su DataFrame de logits
def visualize_image_with_logits(image, logits, num_to_human):
    # Visualizar la imagen
    plt.figure(figsize=(5, 5))
    img = image.cpu().numpy().transpose((1, 2, 0))
    img = img * 0.5 + 0.5  # Desnormalizar
    plt.imshow(np.clip(img, 0, 1))
    plt.axis('off')
    plt.show()

    # Obtener el DataFrame de logits
    df = logits_to_dataframe(logits, num_to_human)
    return df


import numpy as np
from PIL import Image


image_counter = 0

def generate_image_from_variables(variables):
    global image_counter
    image_counter += 1

    num_variables = len(variables)
    # Determinar el número de canales basado en la divisibilidad por 3
    if num_variables % 3 == 0:
        # Configuración para 3 canales (RGB)
        num_channels = 3
        num_pixels = num_variables // num_channels
        image_mode = 'RGB'
    else:
        # Configuración para 1 canal (escala de grises)
        num_channels = 1
        num_pixels = num_variables // num_channels
        image_mode = 'L'

    image_side_length = int(np.sqrt(num_pixels))
    if image_side_length**2 != num_pixels:
        raise ValueError("El número de variables no permite una configuración cuadrada de píxeles.")

    # Reorganizar el array para que tenga la forma correcta según el número de canales
    image_array = variables.reshape((image_side_length, image_side_length, num_channels))

    # Asegurar que los datos estén en el rango correcto [0, 255] y sean enteros
    image_array = np.clip(image_array, 0, 255).astype(np.uint8)

    # Crear una imagen usando PIL y el array
    image = Image.fromarray(image_array.squeeze(), image_mode)  # .squeeze() elimina dimensiones de tamaño 1

    # Guardar la imagen cada x iteraciones o en la primera iteración
    if image_counter % 8349 == 0 or image_counter == 1:
        image_path = f"imagenes/image_{image_counter}.png"  # Definir la ruta del archivo
        image.save(image_path)  # Guardar la imagen en el archivo
        print(f"Image saved: {image_path}")  # Opcional: imprimir la ruta de la imagen guardada

    return image







# Crear el CustomDataset
custom_all_data = CustomDataset(all_data, synset_to_num)

# Crear un DataLoader para iterar sobre los datos
data_loader = DataLoader(custom_all_data, batch_size=1, shuffle=False)  # Batch size de 1 para procesar imagen por imagen



import matplotlib.pyplot as plt
import numpy as np

def get_pixels_from_labelled_images(data_loader, num_to_human, label, num_samples=200):
    count = 0
    pixel_arrays = []  # Lista para almacenar arrays de píxeles de las imágenes que coinciden con la etiqueta
    for images, labels in data_loader:
        for i in range(len(labels)):
            label_text = num_to_human[labels[i].item()]
            if label_text.lower() == label.lower():  # Comprueba si la etiqueta coincide exactamente
                # Visualizar la imagen
                plt.figure(figsize=(5, 5))
                image_permuted = images[i].permute(1, 2, 0)  # Cambiar a [H, W, C]
                plt.imshow(image_permuted)
                plt.title(label_text)
                plt.axis('off')
                plt.show()

                # Convertir a numpy array y aplanar
                pixels = image_permuted.numpy().flatten()
                pixel_arrays.append(pixels*255)

                count += 1
                if count >= num_samples:
                    break
        if count >= num_samples:
            break

    return pixel_arrays

# Llamar a la función para obtener los arrays de píxeles
pixel_arrays = get_pixels_from_labelled_images(data_loader, num_to_human, label="tarantula")

# Imprimir los arrays de píxeles
for i, pixels in enumerate(pixel_arrays):
    pixels=pixels
    print(f"Array de píxeles de la imagen {i+1}:", pixels)









import pandas as pd
import torch
from torchvision import transforms
from torchvision.transforms.functional import to_tensor
from PIL import Image

import matplotlib.pyplot as plt

import matplotlib.pyplot as plt

# Inicializa un contador global
image_display_counter = 0

import os



def wrapped_fitness_function(variables):
# Generar la imagen desde el vector de variables
    pil_image = generate_image_from_variables(variables)

    # Transformar la imagen para el modelo
    image_tensor = transform(pil_image).unsqueeze(0).to(model.device)

    # Obtener logits del modelo
    model.eval()
    with torch.no_grad():
        logits = model(image_tensor).logits.squeeze()  # Asegurarse de que el tensor está bien dimensionado

    # Convertir logits a DataFrame para buscar fácilmente la clase 'tiger'
    logits_df = pd.DataFrame({
        'logit': logits.cpu().numpy(),
        'label': [num_to_human[idx] for idx in range(len(logits))]
    })

    # Buscar el logit de la clase 'tiger'
    # Asumiendo que 'tiger' podría no estar directamente en el string, buscamos subcadenas
    tiger_row = logits_df[logits_df['label'].str.contains('tarantula', case=False, regex=True)]
    if not tiger_row.empty:
        tiger_logit = tiger_row['logit'].iloc[0]
    else:
        print("Tiger class not found, returning a high negative value to handle error.")
        tiger_logit = -1000  # Un valor muy negativo para manejar el caso de que 'tiger' no esté presente

    # Devolver el negativo del logit de "tiger" como valor de fitness
    return -tiger_logit




def convert_to_rgb(image):
    if image.mode != 'RGB':
        return image.convert('RGB')
    return image

transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.Lambda(convert_to_rgb),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


import EDAspy2
from EDAspy2.optimization import UMDAc, EGNA, UnivariateKEDA




# # Configuración de EGNA
egna = UMDAc(
    size_gen=25,
    max_iter=30000,
    dead_iter=25000,
    n_variables=256*256,
    lower_bound=0,
    upper_bound=255,
    alpha=0.5 #,
    # init_data=np.array(pixel_arrays))







# egna = UnivariateKEDA(
#     size_gen=2500,
#     max_iter=100000,
#     dead_iter=50000,
#     n_variables=224*224,
#     lower_bound=0,
#     upper_bound=255,
#     alpha=0.5
# )

# Configuración de EGNA
# egna = EGNA(
#     size_gen=10,
#     max_iter=300,
#     dead_iter=10,
#     n_variables=224*224,
#     lower_bound=0,
#     upper_bound=255,
#     alpha=0.5)

print(egna)
egna.print_parameters()


best_solution = egna.minimize(wrapped_fitness_function)


print(best_solution)





