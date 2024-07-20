

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import gc

from sklearn.datasets import load_iris, load_diabetes, load_digits, load_wine, \
    load_breast_cancer, fetch_california_housing, fetch_covtype
from sklearn.model_selection import train_test_split

from EDAspy2.optimization import EBNA
from EDAspy2.optimization import plot_bn

import networkx as nx

import warnings
warnings.filterwarnings('ignore')
import warnings
warnings.filterwarnings('ignore', message='Probability values don\'t exactly sum to 1.*')
from tqdm import tqdm





import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris, load_diabetes, load_digits, load_wine, load_breast_cancer, fetch_california_housing, fetch_covtype

def load_dataset(name, return_as_pandas=False):
    datasets = {
        'iris': load_iris,
        'diabetes': load_diabetes,
        'digits': load_digits, # es de imagenes pero como son 8x8 lo vamos a dejar como si fuera tabular
        'wine': load_wine,
        'breast_cancer': load_breast_cancer,
        'california_housing': fetch_california_housing,
        'forest_covertypes': fetch_covtype,
        # 'olivetti_faces': fetch_olivetti_faces,
        # 'lfw_people': fetch_lfw_people
    }

    if name not in datasets:
        raise ValueError(f"Dataset '{name}' no disponible.")

    data = datasets[name]()

    X, y = data.data, data.target

    # Dividir el conjunto de datos en entrenamiento, validación y prueba
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.25, random_state=42)

    if return_as_pandas:
        # Combinar los conjuntos en un solo DataFrame
        train_df = pd.DataFrame(X_train)
        train_df['target'] = y_train
        train_df['set'] = 'train'

        val_df = pd.DataFrame(X_val)
        val_df['target'] = y_val
        val_df['set'] = 'validation'

        test_df = pd.DataFrame(X_test)
        test_df['target'] = y_test
        test_df['set'] = 'test'

        full_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
        return full_df

    return X_train, X_val, X_test, y_train, y_val, y_test




def build_model(input_dim, output_dim, hparams):
    class CustomNN(nn.Module):
        def __init__(self, input_dim, output_dim, hparams):
            super(CustomNN, self).__init__()
            self.layers = nn.ModuleList()
            self.dropout_rates = []  # Para almacenar las tasas de dropout

            layer_dims = [int(x) for x in hparams['rec_hidden_layers_config'].split('_')]
            prev_dim = input_dim

            for layer_dim in layer_dims:
                layer = nn.Linear(prev_dim, layer_dim)
                self.layers.append(layer)
                prev_dim = layer_dim

                # Verificar y almacenar la tasa de dropout si se especifica
                if 'dropout' in hparams['regularization']:
                    dropout_rate = float(hparams['regularization'].split('_')[1])
                    self.dropout_rates.append(dropout_rate)
                else:
                    self.dropout_rates.append(0)  # No dropout

            self.output_layer = nn.Linear(prev_dim, output_dim)

        def forward(self, x):
            for i, layer in enumerate(self.layers):
                x = layer(x)
                # Aplicar función de activación común a toda la red
                if hparams['activation'] == 'relu':
                    x = F.relu(x)
                elif hparams['activation'] == 'tanh':
                    x = torch.tanh(x)
                elif hparams['activation'] == 'sigmoid':
                    x = torch.sigmoid(x)
                elif hparams['activation'] == 'leaky_relu':
                    x = F.leaky_relu(x)
                elif hparams['activation'] == 'elu':
                    x = F.elu(x)

                # Aplicar dropout según la configuración
                if self.dropout_rates[i] > 0:
                    x = F.dropout(x, p=self.dropout_rates[i], training=self.training)

            x = self.output_layer(x)
            return F.log_softmax(x, dim=1)

    return CustomNN(input_dim, output_dim, hparams)



def initialize_frequency_for_combined(possible_values):
    frequency = {var: [1/len(possible_values[var])] * len(possible_values[var]) for var in possible_values}
    return frequency




def train_and_evaluate_model(hparams, input_dim, output_dim, X_train, y_train, X_val, y_val):
    # Verificar si CUDA está disponible y configurar el dispositivo
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Construir el modelo y moverlo al dispositivo apropiado (GPU si está disponible)
    model = build_model(input_dim, output_dim, hparams).to(device)

    # Seleccionar el optimizador basado en los hiperparámetros
    if hparams['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=hparams['learning_rate'])
    elif hparams['optimizer'] == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=hparams['learning_rate'])
    elif hparams['optimizer'] == 'momentum':
        # Nota: 'momentum' no es un optimizador en sí, pero se puede implementar con SGD y un valor de momento.
        optimizer = torch.optim.SGD(model.parameters(), lr=hparams['learning_rate'], momentum=0.9)
    elif hparams['optimizer'] == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=hparams['learning_rate'])
    elif hparams['optimizer'] == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=hparams['learning_rate'])

    # Definir el criterio de pérdida
    criterion = nn.CrossEntropyLoss().to(device)

    if isinstance(X_train, np.ndarray):
            X_train = torch.tensor(X_train, dtype=torch.float32).to(device)
    else:
        X_train = X_train.to(device)

    if isinstance(y_train, np.ndarray):
        y_train = torch.tensor(y_train, dtype=torch.int64).to(device)
    else:
        y_train = y_train.to(device)

    if isinstance(X_val, np.ndarray):
        X_val = torch.tensor(X_val, dtype=torch.float32).to(device)
    else:
        X_val = X_val.to(device)

    if isinstance(y_val, np.ndarray):
        y_val = torch.tensor(y_val, dtype=torch.int64).to(device)
    else:
        y_val = y_val.to(device)

    # Preparar los DataLoader para entrenamiento y validación
    train_dataset = TensorDataset(X_train, y_train)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hparams['batch_size'], shuffle=True)

    val_dataset = TensorDataset(X_val, y_val)
    val_loader = DataLoader(dataset=val_dataset, batch_size=hparams['batch_size'], shuffle=False)


    # Entrenamiento del modelo
    model.train()
    for epoch in range(hparams['epochs']):
        for X_batch, y_batch in train_loader:
            optimizer.zero_grad()
            output = model(X_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            optimizer.step()

    # Evaluación del modelo
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            output = model(X_batch)
            _, predicted = torch.max(output.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()

    accuracy = correct / total
    # print(f"Accuracy: {accuracy * 100:.2f}%")


        # Liberar memoria de CUDA
    if device == 'cuda':
        torch.cuda.empty_cache()
        del model
        del X_train, y_train, X_val, y_val
        gc.collect()

    return accuracy








def hyperparameter_search(dataset_name, hyperparameters, size_gen=20, max_iter=20, dead_iter=5,alpha=0.8,output_file=""):
    # Cargar los datos
    X_train, X_val, X_test, y_train, y_val, y_test = load_dataset(dataset_name)

    # Convertir los datos a tensores de PyTorch si aún no lo son
    # Verificar y convertir X_train, X_val, X_test si es necesario
    X_train = torch.tensor(X_train, dtype=torch.float) if not torch.is_tensor(X_train) else X_train
    X_val = torch.tensor(X_val, dtype=torch.float) if not torch.is_tensor(X_val) else X_val
    X_test = torch.tensor(X_test, dtype=torch.float) if not torch.is_tensor(X_test) else X_test

    # Verificar y convertir y_train, y_val, y_test si es necesario
    y_train = torch.tensor(y_train, dtype=torch.long) if not torch.is_tensor(y_train) else y_train
    y_val = torch.tensor(y_val, dtype=torch.long) if not torch.is_tensor(y_val) else y_val
    y_test = torch.tensor(y_test, dtype=torch.long) if not torch.is_tensor(y_test) else y_test

    # Preparar los datos para PyTorch
    input_dim = X_train.shape[1]  # Dimensiones de entrada
    all_labels = torch.cat((y_train, y_val, y_test))  # Ahora todos son tensores de PyTorch
    unique_labels = torch.unique(all_labels)

    # Ajustar las etiquetas si el valor mínimo es 1
    if all_labels.min() == 1:
        y_train -= 1
        y_val -= 1
        y_test -= 1
        print("Las etiquetas han sido ajustadas para empezar en 0.")
        # Recalcular las etiquetas únicas después del ajuste
        all_labels = torch.cat((y_train, y_val, y_test))
        unique_labels = torch.unique(all_labels)

    output_dim = len(unique_labels)  # Determinar la cantidad de clases únicas
    print(unique_labels)

    # Configuración de EBNA (sin cambios)
    possible_values_numeric = {i: hyperparameters[var] for i, var in enumerate(hyperparameters)}
    frequency_numeric = initialize_frequency_for_combined(possible_values_numeric)
    ebna = EBNA(
        size_gen=size_gen,
        max_iter=max_iter,
        dead_iter=dead_iter,
        n_variables=len(possible_values_numeric),
        alpha=alpha,
        possible_values=possible_values_numeric,
        frequency=frequency_numeric
    )

    # Wrapper y ejecución de EBNA (sin cambios)
    def multiKR_cost_wrapper_eda(solution_array):
        hyperparameter_names = list(hyperparameters.keys())
        solution_dict = {name: value for name, value in zip(hyperparameter_names, solution_array)}

        # Conversión de tipos
        solution_dict['learning_rate'] = float(solution_dict['learning_rate'])
        solution_dict['epochs'] = int(solution_dict['epochs'])
        solution_dict['batch_size'] = int(solution_dict['batch_size'])

        # print(f"Los hiperparámetros seleccionados son: {solution_dict}")
        accuracy = train_and_evaluate_model(solution_dict, input_dim, output_dim, X_train, y_train, X_val, y_val)
        return -accuracy  # Negativo de la precisión

    ebna_result = ebna.minimize(multiKR_cost_wrapper_eda)

    # Antes de llamar a ebna.minimize(multiKR_cost_wrapper_eda)
    hyperparameter_names = list(hyperparameters.keys())



    # Suponiendo que hyperparameter_names es una lista de nombres en el mismo orden que el modelo los utiliza
    index_to_name = {i: name for i, name in enumerate(hyperparameter_names)}

    # Obtener la estructura de la red y convertir los índices a nombres
    arcs = ebna.pm.print_structure()  # Asumiendo que esto devuelve una lista de tuplas como [(0, 1), (2, 3), ...]
    arcs_with_names = []

    # Recorre todos los arcos y convierte los índices en nombres
    for arc in arcs:
        try:
            # Convierte los índices a enteros si son cadenas
            start_index = int(arc[0])
            end_index = int(arc[1])

            # Encuentra los nombres correspondientes y los agrega a la lista
            arcs_with_names.append((index_to_name[start_index], index_to_name[end_index]))
        except KeyError as e:
            print(f"Key error: {e}. This index does not exist in the hyperparameter names.")
        except ValueError as e:
            print(f"Value error: {e}. The arc should be a tuple of integers.")


    # Ahora, al generar el gráfico, usa los arcos con nombres en lugar de índices numéricos
    G = nx.DiGraph()
    G.add_nodes_from(hyperparameter_names)
    G.add_edges_from(arcs_with_names)

    # Generar un layout para los nodos si _set_positions no funciona como se espera
    pos = nx.circular_layout(G)

    # Llama a plot_bn pasando 'pos' explícitamente
    plot_bn(
        arcs=arcs_with_names,
        var_names=hyperparameter_names,
        pos=pos,  # Pasa el layout generado como 'pos'
        title="Estructura de la Red Bayesiana", output_file=output_file)

    return ebna_result





def plot_accuracy_evolution(ebna_result, dataset_name=""):
    # Convertir los valores negativos de precisión a positivos, ya que se minimizó el negativo de la precisión
    accuracy_values = [-x for x in ebna_result.history]

    # Crear la figura y los ejes
    plt.figure(figsize=(14, 6))

    # Título y etiquetas
    plt.title('Mejor individuo por generación en '+ str(dataset_name))
    plt.xlabel('Generación')
    plt.ylabel('Accuracy')

    # Corregir el rango del eje x para que comience en 1 y termine en el número de generaciones
    generations = list(range(1, len(accuracy_values) + 1))

    # Graficar la línea que muestra la mejora de la precisión
    plt.plot(generations, accuracy_values, color='b', label='EBNA', marker='o')

    # Añadir una marca de dato en cada punto de la línea
    for i, acc in enumerate(accuracy_values, start=1):
        plt.annotate(f'{acc:.4f}', (i, acc), textcoords="offset points", xytext=(0,10), ha='center')

    # Mostrar la leyenda
    plt.legend()

    # Ajustar los límites del eje x para mejorar la presentación
    plt.xlim(0.5, len(accuracy_values) + 0.5)

    # Mostrar la gráfica
    plt.show()


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Clasificando los conjuntos de datos en tres categorías: multiclase, binario y regresión
datasets_multiclase = [
    'iris',
    'digits',
    'wine',
    'forest_covertypes'
]

datasets_binario = [
    'breast_cancer'
]

datasets_regresion = [
    'diabetes',
    'california_housing'
]

datasets_multiclase, datasets_binario, datasets_regresion


from itertools import product

# Define las opciones de neuronas por capa y el número máximo de capas
neuron_options = ['32','64', '128', '256', '512', '1024', '2048', "4096", "8192"]
max_layers = 7

# Generar todas las posibles configuraciones de capas
layer_configurations = []
for num_layers in range(1, max_layers + 1):
    for combination in product(neuron_options, repeat=num_layers):
        # Convertir la tupla de combinación en una cadena de configuración
        config_string = '_'.join(combination)
        layer_configurations.append(config_string)





valor_inicial = 0.1
learning_rate = [valor_inicial]
for i in range(6):  
    if i % 2 == 0:
        valor_inicial /= 10
    else:  
        valor_inicial /= 2
    learning_rate.append(valor_inicial)

hyperparameters = {
        'rec_hidden_layers_config': layer_configurations,
        'activation': ['relu', 'tanh', 'sigmoid', 'leaky_relu', 'elu'],
        'regularization': ['none', 'dropout_0.1', 'dropout_0.2', 'dropout_0.3', 'dropout_0.4', 'dropout_0.5'],
        'optimizer': ['sgd', 'momentum', 'adam', 'rmsprop', 'adamw'],
        'learning_rate': learning_rate,
        'epochs': list(range(5, 205, 5)),
        'batch_size': [16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    }



dataset_name=datasets_multiclase[3]
print(dataset_name)
ebna_result = hyperparameter_search(dataset_name, hyperparameters, size_gen=40, max_iter=150, dead_iter=10,alpha=0.5, output_file="tresuno.png")
print(ebna_result)
plot_accuracy_evolution(ebna_result, dataset_name)


# dataset_name=datasets_multiclase[3]
# print(dataset_name)
# ebna_result = hyperparameter_search(dataset_name, hyperparameters, size_gen=50, max_iter=150, dead_iter=10,alpha=0.8,output_file="tresdos.png")
# print(ebna_result)
# plot_accuracy_evolution(ebna_result, dataset_name)


