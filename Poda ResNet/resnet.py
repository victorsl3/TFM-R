 
# ### Librerias

from urllib.request import urlopen
from PIL import Image
import timm
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
import os
import copy
import EDAspy2
import sys
sys.path.append("/home/v839/v839190/Hip")
from EDAspy2.optimization import EBNA
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import json
from transformers import AutoImageProcessor, ResNetForImageClassification


root_data="/home/v839/v839190/poda/poda efficientnet/data/"


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

processor = AutoImageProcessor.from_pretrained("microsoft/resnet-50")
model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")


model = model.eval()  # Poner el modelo en modo de evaluación
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

model = model.to(device)


data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)

 
# 
# ## Paso 1: Inicialización
# 
# - **Recopilación de Datos Iniciales**: Evaluar el modelo para obtener un conjunto inicial de métricas de rendimiento que pueden incluir la precisión de la clasificación, la pérdida de la red, o cualquier otra métrica relevante.
# - **Determinación de Importancia de Pesos**: Inicialmente, se puede utilizar un criterio simple como la magnitud del peso o una evaluación de la contribución del peso al gradiente para clasificar los pesos según su importancia.

 
# ### 1.1 Carga de datos


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
batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)


def load_data_to_device(data_loader, device):
    for images, labels in tqdm(data_loader, desc='Loading data to device'):
        images = images.to(device)
        labels = labels.to(device)


# for images, labels in train_loader:
#     images = images.to(device)
#     labels = labels.to(device)
load_data_to_device(train_loader, device)


for images, labels in valid_loader:
    images = images.to(device)
    labels = labels.to(device)


for images, labels in test_loader:
    images = images.to(device)
    labels = labels.to(device)


import matplotlib.pyplot as plt
import numpy as np

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean  # desnormalizar
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title, fontsize=12)
    plt.pause(0.001)  # pause a bit so that plots are updated


# Obtener un batch de imágenes y etiquetas del DataLoader de entrenamiento
dataiter = iter(test_loader)
images, labels = next(dataiter)

# Convertir la etiqueta numérica a descripción legible por humanos
label_text = [num_to_human[label.item()] for label in labels]  # Convertir todas las etiquetas del batch

# Mostrar imágenes y etiquetas
fig, ax = plt.subplots(figsize=(5, 5))  # Ajusta el tamaño según sea necesario

# Visualizar la primera imagen del batch con descripción legible por humanos como título
imshow(images[0], label_text[0])

 
# ### 1.2 Evaluación


# Función para evaluar el modelo
from tqdm import tqdm

def evaluate_model(model, data_loader, device):
    model = model.to(device)

    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in tqdm(data_loader, desc="Evaluating", leave=True):
            # model.to(device)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            # _, predicted = torch.max(outputs.data, 1)
            _, predicted = torch.max(outputs.logits, 1)  # Corregido
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the test images: {accuracy}%')
    return accuracy



# Evaluar el modelo
evaluate_model(model, test_loader, device)

 
# ## 1.3 NTK

 
# ### NTK aproximado




def sample_weight_perturbation(model, epsilon=1e-5, device='cuda'):
    # Esta función crea un diccionario de perturbaciones para los parámetros del modelo
    delta_theta = {}
    for name, param in model.named_parameters():
        if 'weight' in name:  # Solo se consideran los pesos para la perturbación
            perturbation = torch.randn_like(param).to(device) * epsilon
            delta_theta[name] = perturbation
    return delta_theta

def compute_ntk_approx(model, data_loader, device, epsilon=1e-5):
    model.eval()  # Ponemos el modelo en modo de evaluación para desactivar Dropout, etc.
    ntk_approximations = []

    # Asegurarnos de que no calculamos gradientes hasta que sea necesario
    with torch.no_grad():
        for inputs, _ in tqdm(data_loader, desc='Computing NTK Approximation'):
            inputs = inputs.to(device)

            # Activamos los gradientes solo para la sección que los necesita
            with torch.enable_grad():
                outputs_original = model(inputs)
                logits_original = outputs_original if isinstance(outputs_original, torch.Tensor) else outputs_original.logits
                logits_sum = logits_original.sum()  # Suma para poder llamar backward
                logits_sum.backward(retain_graph=True)  # Calculamos gradientes para los pesos originales
                grad_original = torch.cat([param.grad.view(-1) for param in model.parameters() if param.requires_grad])
                model.zero_grad()  # Limpiamos los gradientes para la siguiente pasada

            # Aplicamos la perturbación y calculamos los gradientes con los pesos perturbados
            delta_theta = sample_weight_perturbation(model, epsilon, device)
            for name, param in model.named_parameters():
                if name in delta_theta:
                    param.data.add_(delta_theta[name])

            with torch.enable_grad():
                logits_perturbed = model(inputs).logits
                logits_perturbed.sum().backward()  # Calculamos gradientes para los pesos perturbados
                grad_perturbed = torch.cat([param.grad.view(-1) for param in model.parameters() if param.requires_grad])

            # Restauramos los pesos originales del modelo
            with torch.no_grad():
                for name, param in model.named_parameters():
                    if name in delta_theta:
                        param.data.sub_(delta_theta[name])

            # Calculamos la aproximación de la norma nuclear del NTK para este mini-batch
            ntk_approximation = ((grad_original - grad_perturbed).norm() ** 2) / (epsilon ** 2)
            ntk_approximations.append(ntk_approximation.item())

    ntk_nuclear_norm_approx = sum(ntk_approximations) / len(ntk_approximations)
    return ntk_nuclear_norm_approx

 
# ### NTK de referencia


for name, param in model.named_parameters():
    if not param.requires_grad:
        print(f"Gradient not required for {name}")


# Asegúrate de que 'model', 'valid_loader', y 'device' están definidos y configurados correctamente
ntk_approx = compute_ntk_approx(model, valid_loader, device)


referencia=ntk_approx
referencia

 
# ## Paso 2: Construcción del Modelo Probabilístico
# 
# - **Modelado de la Importancia de los Pesos**: Utilizar los datos recopilados para construir un modelo probabilístico que asocie la importancia de los pesos con su impacto en el rendimiento de la red. Este modelo se actualizará iterativamente para reflejar el aprendizaje adquirido sobre la distribución de la importancia de los pesos.
# - **Aplicación de EDAs**: Implementar un EDA para muestrear y evaluar configuraciones de pesos según el modelo probabilístico.


for name, module in model.named_modules():
    if isinstance(module, torch.nn.Conv2d):
        # Acceder a los pesos
        weights = module.weight.data




def create_pruning_mask(model, param_configuration, exclude_layers_prefixes=['classifier', 'pooler', 'embedder'], exclude_types=['norm']):
    """
    Crea una máscara de poda para el modelo basada en la configuración de parámetros dada.
    Excluye la poda de capas específicas cuyos nombres contienen algún prefijo en exclude_layers_prefixes
    y también excluye los tipos de capas específicos como BatchNorm.
    """
    mask = {}
    idx = 0  # Índice para recorrer la configuración de parámetros
    for name, param in model.named_parameters():
        exclude = False
        for excluded_prefix in exclude_layers_prefixes:
            if excluded_prefix in name:
                exclude = True
                break
        for exclude_type in exclude_types:
            if exclude_type in name and not exclude:
                exclude = True
                break
        if exclude:
            mask[name] = torch.ones_like(param)
        else:
            param_numel = param.numel()
            current_param_config = param_configuration[idx:idx+param_numel]
            mask[name] = torch.tensor(current_param_config, device=param.device).view_as(param)
            idx += param_numel
    return mask

def apply_pruning_mask(model, pruning_mask):
    """
    Aplica la máscara de poda al modelo. Los pesos con un valor de máscara de cero se 'podan',
    y se congela su contribución en entrenamientos futuros asegurando que sus gradientes permanezcan en cero.
    """
    pruned_params = {}
    for name, param in model.named_parameters():
        before_pruning = torch.count_nonzero(param.data)
        param.data.mul_(pruning_mask[name])
        after_pruning = torch.count_nonzero(param.data)
        pruned_count = before_pruning - after_pruning

        path_parts = name.split('.')
        layer_detail = path_parts[1] if len(path_parts) > 1 else 'other'
        subcomponent = path_parts[-1]

        detail_key = f"{layer_detail}.{subcomponent}"
        if detail_key in pruned_params:
            pruned_params[detail_key] += pruned_count
        else:
            pruned_params[detail_key] = pruned_count

        # Añadir un hook para mantener los gradientes en cero para los pesos podados
        if pruning_mask[name].sum() != torch.numel(pruning_mask[name]):  # Verificar si hay algún cero en la máscara
            param.register_hook(lambda grad, mask=pruning_mask[name]: grad * mask)

    print("Resumen detallado de parámetros podados:")
    for detail_key, count in sorted(pruned_params.items()):
        print(f"{detail_key}: {count} parámetros podados")

    return model

# Ajuste de n_variables para cubrir todos los parámetros del modelo
n_variables = sum(p.numel() for p in model.parameters())

import copy
original_model = copy.deepcopy(model)  # Hace una copia profunda del modelo original
possible_values = [[0, 1] for _ in range(n_variables)]
# frequency = [[0.999, 0.001] for _ in range(n_variables)]
frequency = [[0.001, 0.999] for _ in range(n_variables)]
#frequency = [[0.5, 0.5] for _ in range(n_variables)]



n_variables_total = sum(p.numel() for p in model.parameters())
n_variables_total



def evaluate_accuracy_loss(model, data_loader, device, criterion):
    model.eval()  # Cambia el modelo a modo de evaluación
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        # Utiliza tqdm para visualizar el progreso
        for inputs, targets in tqdm(data_loader, desc="Evaluación", leave=True):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs
            loss = criterion(logits, targets)  # Utiliza los logits aquí
            total_loss += loss.item()
            _, predicted = logits.max(1)  # Usa logits para obtener las predicciones
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)

    # Calcula la precisión y la pérdida promedio
    if total > 0:
        accuracy = 100 * correct / total
        avg_loss = total_loss / len(data_loader)
        return avg_loss, accuracy
    else:
        return 0, 0  # Devuelve 0 para evitar errores en caso de que no se procesen datos




def train_model(model, train_loader, valid_loader, device, epochs, lr, pruning_mask=None, early_stopping_patience=3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=1, verbose=True)

    train_losses = []
    train_accuracies = []
    valid_losses = []
    valid_accuracies = []

    best_valid_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0
        total = 0
        # Envuelve el cargador de entrenamiento con tqdm para la barra de progreso
        train_progress_bar = tqdm(train_loader, desc=f'Training Epoch {epoch + 1}/{epochs}', leave=False)
        for inputs, targets in train_progress_bar:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = model(inputs)
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs

            loss = criterion(logits, targets)
            loss.backward()

            if pruning_mask is not None:
                for name, param in model.named_parameters():
                    if 'weight' in name:
                        param.grad.data.mul_(pruning_mask[name])

            optimizer.step()
            epoch_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            # Actualiza la barra de progreso de tqdm con la última pérdida y precisión
            train_progress_bar.set_postfix(loss=loss.item(), accuracy=f"{100. * correct / total:.2f}%")

        train_losses.append(epoch_loss / len(train_loader))
        train_accuracies.append(100. * correct / total)

        # Evaluar en el conjunto de validación
        valid_loss, valid_accuracy = evaluate_accuracy_loss(model, valid_loader, device, criterion)
        valid_losses.append(valid_loss)
        valid_accuracies.append(valid_accuracy)
        
        # Actualizar el scheduler con la pérdida de validación
        scheduler.step(valid_loss)
        
        print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_losses[-1]:.4f}, Train Acc: {train_accuracies[-1]:.2f}%, Valid Loss: {valid_losses[-1]:.4f}, Valid Acc: {valid_accuracies[-1]:.2f}%")

        # Verificar early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    return train_losses, train_accuracies, valid_losses, valid_accuracies

def get_next_individual_id(filename):
    try:
        with open(filename, 'r') as f:
            for i, l in enumerate(f):
                pass
        return i + 1
    except FileNotFoundError:
        return 0


import copy
import json
import torch

def fitness_function(original_model, train_loader, valid_loader, device, config, ntk_ref, epochs=3, lr=0.001):
    # Archivo para guardar el contador de ejecuciones
    counter_filename = '/home/v839/v839190/poda/poda efficientnet/execution_count.txt'

    # Leer el contador actual o inicializarlo si el archivo no existe
    try:
        with open(counter_filename, 'r') as file:
            execution_count = int(file.read().strip())
    except FileNotFoundError:
        execution_count = 0

    # Incrementar el contador
    execution_count += 1
    print(f"Ejecución número: {execution_count}")
    # Escribir el nuevo valor del contador en el archivo
    with open(counter_filename, 'w') as file:
        file.write(str(execution_count))

    # Mover el modelo original fuera del dispositivo para ahorrar memoria
    original_model.cpu()

    # Clonar y preparar el modelo para la poda y el entrenamiento
    cloned_model = copy.deepcopy(original_model)
    cloned_model.to(device)

    # Crear y aplicar la máscara de poda basada en la configuración
    pruning_mask = create_pruning_mask(cloned_model, config)
    cloned_model = apply_pruning_mask(cloned_model, pruning_mask)

    # Decidir si entrenar o rellenar con ceros basándose en el contador
    if execution_count % 20 == 0:
        # Entrenar el modelo y obtener las métricas
        train_losses, train_accuracies, valid_losses, valid_accuracies = train_model(
            cloned_model, train_loader, valid_loader, device, epochs, lr, pruning_mask
        )
    else:
        # Rellenar los arrays con ceros si no es momento de entrenar
        train_losses, train_accuracies, valid_losses, valid_accuracies = ([0], [0], [0], [0])

    # Calcular la norma nuclear aproximada del NTK después del entrenamiento
    ntk_approx = compute_ntk_approx(cloned_model, valid_loader, device)

    # Calcula la diferencia absoluta con la norma nuclear de referencia del NTK
    cost = abs(ntk_ref - ntk_approx) 

    # Mover el modelo podado de vuelta a la CPU para liberar memoria GPU
    cloned_model.cpu()
    # Limpia la caché de la GPU si es necesario
    torch.cuda.empty_cache()

    # Obtener el ID del individuo basándose en el número de registros en el archivo
    filename = '/home/v839/v839190/poda/poda efficientnet/individual_info.txt'
    individual_id = get_next_individual_id(filename)

    # Información del individuo
    individual_info = {
        'id': individual_id,
        'number_of_parameters': sum(p.numel() for p in cloned_model.parameters()),
        'train_loss': train_losses,
        'train_accuracy': train_accuracies,
        'validation_loss': valid_losses,
        'validation_accuracy': valid_accuracies,
        'ntk_nuclear_norm': ntk_approx,
        'ntk_difference': cost
    }

    # Registrar información del individuo en archivo de texto
    with open(filename, 'a') as f:
        f.write(json.dumps(individual_info) + '\n')

    return cost


for name, param in model.named_parameters():
    if param.requires_grad == False:
        print(name, "has requires_grad set to False")



import EDAspy2
from EDAspy2.optimization import UMDAd, UnivariateKEDA


umda = UMDAd(
    size_gen=50,
    max_iter=250,
    dead_iter=150,
    n_variables=n_variables,
    alpha=0.5
)

# umda = UnivariateKEDA(
#     size_gen=50,
#     max_iter=250,
#     dead_iter=40,
#     n_variables=n_variables,
#     alpha=0.8
# )


# from EDAspy2.optimization import EBNA
# umda = EBNA(
#     size_gen=20,
#     max_iter=120,
#     dead_iter=8,
#     n_variables=n_variables,
#     alpha=0.9,
#     possible_values=possible_values,
#     frequency=frequency

# )

# Definir una función lambda que envuelva la llamada a fitness_function con los parámetros necesarios
wrapped_fitness_function = lambda config: fitness_function(
    original_model,
    train_loader,
    valid_loader,
    device,
    config,  # config se pasa directamente desde el eda
    referencia,
    epochs=10,
    lr=0.001
)

# Ejecutar EDA con la función lambda
best_solution = umda.minimize(wrapped_fitness_function)




