import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objs as go
from plotly.offline import init_notebook_mode, iplot
import random


# Inicializar las variables globales fuera de las funciones


def read_item_index_to_entity_id_file():
    item_index_old2new = {}
    entity_id2index = {}
    relation_id2index = {}
    file_path = './MKR-data/item_index2entity_id.txt'
    # print(f'Reading item index to entity id file: {file_path} ...')
    with open(file_path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            item_index, satori_id = line.strip().split('\t')
            item_index_old2new[item_index] = i  
            entity_id2index[satori_id] = i
    return item_index_old2new, entity_id2index



def convert_rating(item_index_old2new):
    file_path = './MKR-data/BX-Book-Ratings.csv'
    # print(f'Reading rating file: {file_path} ...')
    item_set = set(item_index_old2new.values())
    user_pos_ratings, user_neg_ratings = {}, {}

    with open(file_path, 'r', encoding='utf-8') as file:
        next(file)  # Skip header
        for line in file:
            parts = line.strip().split(';')
            user_index_old = parts[0].strip('"')
            item_index_old = parts[1].strip('"')
            rating = float(parts[2].strip('"'))
            
            # Actualizar para considerar calificaciones de 5 a 10 como positivas
            if item_index_old in item_index_old2new:
                item_index = item_index_old2new[item_index_old]
                if rating >= 5:  # Calificaciones de 5 a 10 como positivas
                    user_pos_ratings.setdefault(user_index_old, set()).add(item_index)
                else:  # Calificaciones de 0 a 4 como negativas
                    user_neg_ratings.setdefault(user_index_old, set()).add(item_index)

    write_final_ratings(user_pos_ratings, user_neg_ratings, item_set)


def write_final_ratings(user_pos_ratings, user_neg_ratings, item_set):
    output_file = './MKR-data/ratings_final.txt'
    # print(f'Converting rating file and writing to {output_file} ...')
    with open(output_file, 'w', encoding='utf-8') as writer:
        user_cnt = 0
        user_index_old2new = {}
        for user_index_old, pos_item_set in user_pos_ratings.items():
            if user_index_old not in user_index_old2new:
                user_index_old2new[user_index_old] = user_cnt
                user_cnt += 1
            user_index = user_index_old2new[user_index_old]
            for item in pos_item_set:
                writer.write(f'{user_index}\t{item}\t1\n')
            for item in user_neg_ratings.get(user_index_old, set()):
                writer.write(f'{user_index}\t{item}\t0\n')
    # print(f'Number of users: {len(user_index_old2new)}')
    # print(f'Number of items: {len(item_set)}')

def write_negative_samples(writer, user_index, pos_item_set, neg_item_set, item_set):
    unwatched_set = item_set - pos_item_set - neg_item_set
    for item in np.random.choice(list(unwatched_set), size=len(pos_item_set), replace=False):
        writer.write(f'{user_index}\t{item}\t0\n')



def convert_kg():
    entity_id2index = {}
    relation_id2index = {}
    file_path = './MKR-data/kg.txt'
    # print(f'Converting KG file: {file_path} ...')
    entity_cnt = 0  # Contador para asignar índices únicos a las entidades
    relation_cnt = 0  # Contador para asignar índices únicos a las relaciones
    output_file = './MKR-data/kg_final.txt'

    # Abrir el archivo de entrada y el archivo de salida
    with open(file_path, 'r', encoding='utf-8') as file, open(output_file, 'w', encoding='utf-8') as writer:
        for line in file:
            head_old, relation_old, tail_old = line.strip().split('\t')

            # Asignar un nuevo índice a la entidad de la cabeza si es necesario
            if head_old not in entity_id2index:
                entity_id2index[head_old] = entity_cnt
                entity_cnt += 1
            head = entity_id2index[head_old]

            # Asignar un nuevo índice a la entidad de la cola si es necesario
            if tail_old not in entity_id2index:
                entity_id2index[tail_old] = entity_cnt
                entity_cnt += 1
            tail = entity_id2index[tail_old]

            # Asignar un nuevo índice a la relación si es necesario
            if relation_old not in relation_id2index:
                relation_id2index[relation_old] = relation_cnt
                relation_cnt += 1
            relation = relation_id2index[relation_old]

            # Escribir el triple convertido al archivo de salida
            writer.write(f'{head}\t{relation}\t{tail}\n')

    # Imprimir resumen de la conversión
    # print(f'Number of entities (including items): {entity_cnt}')
    # print(f'Number of relations: {relation_cnt}')

    # Retornar los mapeos de entidades y relaciones para su uso posterior
    return entity_id2index, relation_id2index



class TrainSet(Dataset):
    """
    Clase para crear un conjunto de datos de entrenamiento.
    """
    def __init__(self, data):
        self.user = torch.LongTensor(data[:, 0])
        self.item = torch.LongTensor(data[:, 1])
        self.target = torch.FloatTensor(data[:, 2])

    def __getitem__(self, index):
        return self.user[index], self.item[index], self.target[index]

    def __len__(self):
        return len(self.target)
    


def dataset_split(ratings, eval_ratio=0.2, test_ratio=0.2):
    """
    Divide el conjunto de datos en entrenamiento, evaluación y prueba.
    
    Parámetros:
    - ratings: Un array de numpy con las calificaciones.
    - eval_ratio: Proporción del conjunto de evaluación.
    - test_ratio: Proporción del conjunto de prueba.
    
    Retorna:
    - Tres conjuntos de datos: entrenamiento, evaluación y prueba.
    """
    n_ratings = ratings.shape[0]
    indices = np.arange(n_ratings)
    np.random.shuffle(indices)
    
    eval_size = int(n_ratings * eval_ratio)
    test_size = int(n_ratings * test_ratio)
    
    eval_indices = indices[:eval_size]
    test_indices = indices[eval_size:eval_size+test_size]
    train_indices = indices[eval_size+test_size:]
    
    return ratings[train_indices], ratings[eval_indices], ratings[test_indices]


# Verificar y resumir los conjuntos de datos
def summarize_dataset(dataset):
    positive_count = sum(dataset.target.numpy())
    total_count = len(dataset)
    print(f"Total examples: {total_count}, Positive examples: {positive_count}, Negative examples: {total_count - positive_count}")
    print(f"Positive proportion: {positive_count / total_count:.2f}")



import plotly.graph_objects as go
from plotly.offline import iplot, init_notebook_mode
import networkx as nx



import plotly.graph_objects as go
from plotly.offline import init_notebook_mode, iplot
import networkx as nx

def visualize_kg_from_file(file_path, entity_id2index, index_to_relation_name, relation_to_entity_types, num_triples_to_plot=50, filter_relation_type=None, min_relations_per_node=None, max_relations_per_node=None):
    """
    Visualiza un subconjunto del grafo de conocimiento desde un archivo de manera interactiva con Plotly, 
    aplicando filtros por tipo de relación y número de relaciones por nodo.

    Args:
    - file_path (str): Ruta al archivo que contiene los triples del grafo de conocimiento.
    - entity_id2index (dict): Diccionario mapeando índices de entidades a IDs/nombres de entidades.
    - index_to_relation_name (dict): Diccionario mapeando índices de relaciones a nombres de relaciones.
    - relation_to_entity_types (dict): Diccionario mapeando nombres de relaciones a tuplas de tipos de entidad (inicio, fin).
    - num_triples_to_plot (int): Número de triples a visualizar.
    - filter_relation_type (str, optional): Tipo de relación para filtrar la visualización.
    - min_relations_per_node (int, optional): Número mínimo de relaciones por nodo para su inclusión.
    - max_relations_per_node (int, optional): Número máximo de relaciones por nodo para su inclusión.
    """
    G = nx.DiGraph()
    relation_count = {}

    # Leer el archivo y construir el grafo
    with open(file_path, 'r', encoding='utf-8') as kg_file:
        for i, line in enumerate(kg_file):
            if i >= num_triples_to_plot:
                break
            head, relation_index, tail = line.strip().split('\t')
            relation_name = index_to_relation_name[int(relation_index)]
            # Aplicar filtro de tipo de relación si se especifica
            if filter_relation_type and relation_name != filter_relation_type:
                continue
            head_type, tail_type = relation_to_entity_types[relation_name]
            # Construir etiquetas para nodos y aristas
            head_label = f"{entity_id2index.get(head, head)} ({head_type})"
            tail_label = f"{entity_id2index.get(tail, tail)} ({tail_type})"
            relation_label = relation_name
            G.add_edge(head_label, tail_label, label=relation_label)
            # Contabilizar relaciones para cada nodo
            relation_count[head_label] = relation_count.get(head_label, 0) + 1
            relation_count[tail_label] = relation_count.get(tail_label, 0) + 1

    # Aplicar filtros de número de relaciones por nodo
    nodes_to_remove = [node for node, count in relation_count.items() if (min_relations_per_node and count < min_relations_per_node) or (max_relations_per_node and count > max_relations_per_node)]
    G.remove_nodes_from(nodes_to_remove)

    pos = nx.spring_layout(G)  # Posición de los nodos

    # Creación de trazas para nodos y aristas
    edge_trace, node_trace = create_traces_for_visualization(G, pos)

    # Configuración e inicialización de la visualización interactiva
    init_notebook_mode(connected=True)
    fig = go.Figure(data=[edge_trace, node_trace], layout=go.Layout(showlegend=False, hovermode='closest', margin=dict(b=20, l=5, r=5, t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False), yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)))
    iplot(fig)


def create_traces_for_visualization(G, pos):
    # Colores para diferentes tipos de entidades
    type_colors = {'book': '#43a2ca', 'author': '#7bccc4', 'genre': '#bae4bc', 'date': '#a8ddb5', 'series': '#ccebc5', 'work': '#f0f9e8', 'publisher': '#feb24c', 'translation': '#fdae61', 'subject': '#f46d43', 'previous_work': '#d53e4f', 'unknown': '#999999'}
    
    # Crear trazas de aristas
    edge_x = []
    edge_y = []
    edge_text = []  # Para almacenar el nombre del tipo de relación
    for edge in G.edges(data=True):
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])
        # Añadir el nombre del tipo de relación a la traza de la arista
        edge_text.append(edge[2]['label'])

    edge_trace = go.Scatter(
        x=edge_x, y=edge_y,
        line=dict(width=1, color='#888'),
        hoverinfo='text',
        mode='lines',
        text=edge_text,  # Asignar el texto de la arista
    )

    # Crear trazas de nodos
    node_x = []
    node_y = []
    node_text = []  # Para almacenar información del nodo
    node_marker = dict(
        showscale=False,  # No mostramos escala de color para los nodos
        colorscale='YlGnBu',
        size=10,
        color=[],  # Colores de los nodos
        line_width=2)

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(node)  # Podrías añadir más información del nodo aquí

        # Extraer el tipo de entidad del nodo
        entity_type = node.split(' ')[-1].strip('()')  # Asumiendo que el tipo de entidad está al final entre paréntesis
        node_marker['color'].append(type_colors.get(entity_type, type_colors['unknown']))

    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers',
        hoverinfo='text',
        text=node_text,
        marker=node_marker)

    return edge_trace, node_trace



# Función para generar mapeo de ID de entidad a tipo
def generate_entity_to_type_mapping(file_path, relation_to_entity_types):
    entity_to_type = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            head, relation, tail = line.strip().split('\t')
            head_type, tail_type = relation_to_entity_types.get(relation, ('unknown', 'unknown'))
            entity_to_type[head] = head_type
            entity_to_type[tail] = tail_type
    return entity_to_type


import ipycytoscape
import networkx as nx
import json

import ipycytoscape
import networkx as nx

def visualize_kg_with_ipycytoscape(file_path, entity_id2index, index_to_relation_name, relation_to_entity_types, num_triples_to_plot=50):
    G = nx.DiGraph()
    entity_types = {}  # Mapeo de entidades a tipos para el estilo

    with open(file_path, 'r', encoding='utf-8') as kg_file:
        for i, line in enumerate(kg_file):
            if i >= num_triples_to_plot:
                break
            head, relation_index, tail = line.strip().split('\t')
            relation_name = index_to_relation_name[int(relation_index)]  # Convertimos el índice a nombre de relación
            head_type, tail_type = relation_to_entity_types[relation_name]  # Obtenemos los tipos de entidades basados en la relación
            
            # Asignamos los tipos a las entidades
            entity_types[head] = head_type
            entity_types[tail] = tail_type

            # Obtenemos las etiquetas para la visualización
            head_label = entity_id2index.get(int(head), f'Entity{head}')
            tail_label = entity_id2index.get(int(tail), f'Entity{tail}')
            relation_label = relation_name  # Usamos el nombre de la relación directamente
            
            # Añadimos los nodos y aristas al grafo
            G.add_edge(head_label, tail_label, label=relation_label)

    # Convertimos el grafo de NetworkX a formato de datos de Cytoscape
    cytoscape_graph = nx.cytoscape_data(G)['elements']

    # Creamos el widget de Cytoscape y cargamos los datos del grafo
    cyto_graph = ipycytoscape.CytoscapeWidget()
    cyto_graph.graph.add_graph_from_json(cytoscape_graph)

    # Definimos los estilos para nodos y aristas
    styles = [
        {
            'selector': 'node',
            'style': {
                'background-color': 'data(typeColor)',
                'label': 'data(id)',
                'text-valign': 'center',
                'color': 'white',
                'text-outline-width': '2px',
                'text-outline-color': '#888',
                'font-size': '10px'
            }
        },
        {
            'selector': 'edge',
            'style': {
                'width': 2,
                'line-color': '#9dbaea',
                'target-arrow-shape': 'triangle',
                'target-arrow-color': '#9dbaea'
            }
        }
    ]

    # Asignamos colores a los nodos según sus tipos
    for node in cyto_graph.graph.nodes:
        node_id_str = node.data['id']
        # Asegúrate de que node_id corresponda a las claves en entity_types
        node_id = node_id_str if node_id_str in entity_types else None  # Ajusta esta línea según tu mapeo
        node_type = entity_types.get(node_id, 'unknown')
        # Asigna colores según el tipo de entidad
        if node_type == 'book':
            node.data['typeColor'] = '#43a2ca'
        elif node_type == 'author':
            node.data['typeColor'] = '#7bccc4'



    # Aplicamos los estilos y la disposición del layout
    cyto_graph.set_style(styles)
    cyto_graph.set_layout(name='cola', nodeSpacing=10, edgeLengthVal=10, infinite=True)

    return cyto_graph





import torch
import networkx as nx


def load_kg_and_create_edge_index(entity_id2index, relation_id2index):
    file_path='./MKR-data/kg.txt'
    edge_list = []
    relation_list = []  # Lista para almacenar las relaciones correspondientes a cada arista

    with open(file_path, 'r', encoding='utf-8') as kg_file:
        for line in kg_file:
            head, relation, tail = line.strip().split('\t')

            # Verificar si head, relation y tail están en los mapeos
            if head in entity_id2index and tail in entity_id2index and relation in relation_id2index:
                head_idx = entity_id2index[head]
                tail_idx = entity_id2index[tail]
                relation_idx = relation_id2index[relation]  # Obtener el índice del tipo de relación
                edge_list.append((head_idx, tail_idx))
                relation_list.append(relation_idx)  # Añadir el índice de la relación
            else:
                print(f"Entidad o relación faltante: {head}, {relation}, {tail}")

    edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
    relation_index = torch.tensor(relation_list, dtype=torch.long)  # Tensor para índices de relaciones

    print(f"Tamaño de edge_index: {edge_index.size()}")
    print(f"Número de aristas: {len(edge_list)}")
    return edge_index, relation_index




def generate_kg_negative_samples(kg_data, num_entities, num_samples=1):
    """
    Genera ejemplos negativos para el conjunto de entrenamiento del KG.
    
    Parámetros:
    - kg_data: Los datos del KG, que contienen tripletas positivas (head, relation, tail).
    - num_entities: El número total de entidades únicas en el dataset del KG.
    - num_samples: El número de ejemplos negativos a generar por cada ejemplo positivo.
    
    Retorna:
    - Un nuevo conjunto de datos que incluye los ejemplos negativos.
    """
    new_data = []
    for head, relation, tail in kg_data:
        # Añadir la tripleta positiva original
        new_data.append([head, relation, tail, 1])  # Agregar un indicador positivo
        for _ in range(num_samples):
            negative_tail = random.randint(0, num_entities-1)
            # Asegurarse de que la entidad de cola negativa no sea igual a la entidad de cola positiva
            while negative_tail == tail:
                negative_tail = random.randint(0, num_entities-1)
            new_data.append([head, relation, negative_tail, 0])  # Agregar como tripleta negativa con indicador negativo
    return np.array(new_data)


# class KGTrainSet(Dataset):
#     """
#     Clase para crear un conjunto de datos de entrenamiento para el KG, incluyendo tripletas positivas y negativas.
#     """
#     def __init__(self, data):
#         # Espera 'data' como un array donde cada fila es [head, relation, tail, label],
#         # y 'label' indica si es una tripla positiva (1) o negativa (0).
#         self.data = data

#     def __getitem__(self, index):
#         head, relation, tail, label = self.data[index]
#         return torch.tensor(head, dtype=torch.long), torch.tensor(relation, dtype=torch.long), torch.tensor(tail, dtype=torch.long), torch.tensor(label, dtype=torch.float)

#     def __len__(self):
#         return len(self.data)
    

class KGTrainSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        head, relation, tail = self.data[index]
        return torch.tensor(head, dtype=torch.long), torch.tensor(relation, dtype=torch.long), torch.tensor(tail, dtype=torch.long)

    def __len__(self):
        return len(self.data)


import numpy as np
import os

def adapt_and_split_kg_data(entity_id2index, kg_file_path, output_path, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    # Leer líneas del archivo kg.txt
    with open(kg_file_path, 'r') as file:
        lines = file.readlines()

    # Adaptar las líneas directamente sin cambiar 'relation' ya que ya está en el formato correcto
    adapted_lines = []
    for line in lines:
        head, relation, tail = line.strip().split('\t')
        head_id = list(entity_id2index.keys())[list(entity_id2index.values()).index(int(head))]
        tail_id = list(entity_id2index.keys())[list(entity_id2index.values()).index(int(tail))]

        # Usar 'relation' directamente
        adapted_lines.append(f'{head_id}\t{relation}\t{tail_id}\n')

    # Mezclar líneas aleatoriamente
    np.random.shuffle(adapted_lines)

    # Calcular el número de datos para cada conjunto
    total_lines = len(adapted_lines)
    num_train = int(total_lines * train_ratio)
    num_valid = int(total_lines * valid_ratio)

    # Dividir los datos adaptados
    train_lines = adapted_lines[:num_train]
    valid_lines = adapted_lines[num_train:num_train + num_valid]
    test_lines = adapted_lines[num_train + num_valid:]

    # Guardar los conjuntos en archivos separados
    for subset_name, subset_lines in zip(['train', 'valid', 'test'], [train_lines, valid_lines, test_lines]):
        with open(os.path.join(output_path, f'custom_dataset-{subset_name}.txt'), 'w') as output_file:
            output_file.writelines(subset_lines)





def adapt_and_split_kg_data_with_slashes(entity_id2index, relation_to_entity_types, kg_file_path, output_path, train_ratio=0.7, valid_ratio=0.15, test_ratio=0.15):
    # Leer líneas del archivo kg.txt
    with open(kg_file_path, 'r') as file:
        lines = file.readlines()

    # Adaptar las líneas con barras "/"
    adapted_lines = []
    for line in lines:
        head, relation, tail = line.strip().split('\t')
        head_type, tail_type = relation_to_entity_types[relation]
        head_id = f"/{head_type}/{list(entity_id2index.keys())[list(entity_id2index.values()).index(int(head))]}"
        tail_id = f"/{tail_type}/{list(entity_id2index.keys())[list(entity_id2index.values()).index(int(tail))]}"

        adapted_lines.append(f'{head_id}\t{relation}\t{tail_id}\n')
        
    # Mezclar líneas aleatoriamente
    np.random.shuffle(adapted_lines)

    # Calcular el número de datos para cada conjunto
    total_lines = len(adapted_lines)
    num_train = int(total_lines * train_ratio)
    num_valid = int(total_lines * valid_ratio)

    # Dividir los datos adaptados
    train_lines = adapted_lines[:num_train]
    valid_lines = adapted_lines[num_train:num_train + num_valid]
    test_lines = adapted_lines[num_train + num_valid:]

    # Guardar los conjuntos en archivos separados
    for subset_name, subset_lines in zip(['train', 'valid', 'test'], [train_lines, valid_lines, test_lines]):
        with open(os.path.join(output_path, f'custom_dataset-{subset_name}.txt'), 'w') as output_file:
            output_file.writelines(subset_lines)

