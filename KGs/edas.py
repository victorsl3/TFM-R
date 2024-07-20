from custom_mkr import MultiKR
from limpieza_datos import read_item_index_to_entity_id_file, convert_rating, convert_kg, visualize_kg_from_file, generate_entity_to_type_mapping, dataset_split, DataLoader, TrainSet, summarize_dataset
from torch.utils.data import Dataset, DataLoader
from train_and_evaluate import train_epoch, evaluate_model, train_and_evaluate
from EDAspy.optimization import EBNA
import numpy as np
import torch
from KGs import *
from sklearn.metrics import roc_auc_score

def setup_model_and_dataloaders(solution):

    # Procesar los datos y capturar los mapeos
    _, entity_id2index = read_item_index_to_entity_id_file()
    convert_rating(_)
    entity_id2index, relation_id2index = convert_kg()



    # Asumiendo que solution es un diccionario con el formato adecuado
    batch_size = solution['batch_size']
    lr = solution['lr']
    embed_dim = solution['embed_dim']
    hidden_layers_config = solution['hidden_layers_config']
    dropout_rate = solution['dropout_rate']


    # Convertir la configuración de capas ocultas en una lista de enteros
    hidden_layers = [int(size) for size in hidden_layers_config.split('_')]
    n_layer = len(hidden_layers)  # Calcula n_layer basado en la longitud de hidden_layers

    # Preparar los datos para el entrenamiento y la evaluación
    ratings = np.loadtxt('./MKR-data/ratings_final.txt', dtype=np.int32)
    train_data, eval_data, test_data = dataset_split(ratings)

    train_loader = DataLoader(TrainSet(train_data), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TrainSet(eval_data), batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(TrainSet(test_data), batch_size=batch_size, shuffle=False)


    # Parámetros para MultiKR
    user_num = len(np.unique(ratings[:, 0]))  # Número de usuarios únicos
    item_num = len(_)  # Número de ítems únicos
    entity_num = len(entity_id2index)  # Número de entidades únicas
    relation_num = len(relation_id2index)  # Número de relaciones únicas

    # Inicializar modelo con la nueva estructura de capas ocultas y pasar n_layer
    model = MultiKR(user_num, item_num, entity_num, relation_num, n_layer=int(n_layer), embed_dim=int(embed_dim), hidden_layers=hidden_layers, dropouts=[float(dropout_rate)]* len(hidden_layers), output_rec=1)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))

    return model, optimizer, train_loader, val_loader, eval_loader




def multiKR_cost_function(solution, type="rec"):
    """
    Función de costo adaptada para utilizar setup_model_and_dataloaders.
    """
    # Instanciar modelo, optimizador y dataloaders con la solución actual
    model, optimizer, train_loader, val_loader, eval_loader = setup_model_and_dataloaders(solution)

    loss_function = torch.nn.BCEWithLogitsLoss()

    epochs = solution['epochs']  # Extrae epochs del diccionario de solución

    # Entrenar el modelo
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, epoch, "rec")

    # Evaluar el modelo usando val_loader para no filtrar información del conjunto de test
    auc = evaluate_model(model, val_loader, epochs - 1)

    # Devolver el negativo de AUC para minimización
    return -auc




def decode_solution(solution_array):
    return {
        'batch_size': int(solution_array[0]),
        'lr': solution_array[1],
        'embed_dim': int(solution_array[2]),
        'hidden_layers_config': solution_array[3],
        'dropout_rate': solution_array[4],
        'output_rec': int(solution_array[5]),
        'epochs': int(solution_array[6])  # Asume el índice correcto para epochs
    }




def multiKR_cost_wrapper(solution_array):
    # Convierte el array en un diccionario usando decode_solution
    solution_dict = decode_solution(solution_array)
    print(f"Los hiperparámetros seleccionados son: {solution_dict}")
    # Llama a multiKR_cost_function pasando el diccionario decodificado
    return multiKR_cost_function(solution_dict)


def define_initial_frequency(variables, possible_values):
    frequency = {var: [1/len(possible_values[var])] * len(possible_values[var]) for var in variables}
    return frequency


def define_variables_for_multiKR():
    variables = ['batch_size', 'lr', 'embed_dim', 'hidden_layers_config', 'dropout_rate', 'output_rec', 'epochs']
    return variables

def define_possible_values_for_multiKR():
    possible_values = {
        'batch_size': [32, 64, 128],
        'lr': [0.01, 0.001, 0.0001],
        'embed_dim': [64, 128, 256],
        'hidden_layers_config': [
            '64', '128', '256',  # Configuraciones de una sola capa
            '64_64', '128_128', '256_256',  # Configuraciones de dos capas iguales
            '64_128', '128_256',  # Configuraciones de dos capas crecientes
            '128_64', '256_128',  # Configuraciones de dos capas decrecientes
            '64_128_256', '256_128_64',  # Configuraciones de tres capas
            '64_64_64', '128_128_128', '256_256_256',  # Configuraciones de tres capas iguales
        ],
        'dropout_rate': [0.5, 0.3, 0.1],
        'output_rec': [1],
        'epochs': [5, 10, 15]  # Agrega los valores deseados para epochs
    }
    return possible_values




##########################################################################################################################################################

def define_variables_for_KG_and_rec():
    variables_kg = {
        'kg_batch_size': [64, 128, 256, 512],
        'kg_epochs': [1,1], #[20, 50, 100, 200],
        'kg_learning_rate': [0.00001, 0.0001, 0.001, 0.01],
        'kg_model_name': [
            'AcrE', 'ANALOGY',  'ConvKB', 'CP',
            'DistMult', 'HoLE', 'HypER', 'InteractE', 'KG2E', 'MuRP', 'NTN',
             'ProjE_pointwise', 'QuatE', 'Rescal', 'RotatE',
            'SimplE', 'SimplE_ignr', 'SLM', 'SME_BL', 'TransD', 'TransE',
            'TransH', 'TransM', 'TransR', 'TuckER'
            # 'SME', 'ConvE','OctonionE','Complex', 'ComplexN3',
        ],
        'lmbda': [0.01, 0.1, 1.0],
        'margin': [0, 0.5, 1, 2, 5, 10],
        'optimizer': ['adam', 'sgd', 'rms', 'riemannian'],
        'sampling': ['uniform', 'bern'],
        'neg_rate': [1, 2, 3],
        'rel_hidden_size': [8, 16, 32, 64, 128, 256],
        'l1_flag': [True, False],
        'alpha': [0.01, 0.1, 0.5, 1.0],
        'label_smoothing': [0.0, 0.1, 0.2, 0.3],
    }

    variables_rec = {
        'rec_batch_size': [32, 64, 128],
        'rec_lr': [0.01, 0.001, 0.0001],
        'rec_embed_dim': [64, 128, 256],
        'rec_hidden_layers_config': [
            '64', '128', '256', '64_64', '128_128', '256_256', '64_128', '128_256',
            '128_64', '256_128', '64_128_256', '256_128_64', '64_64_64', '128_128_128', '256_256_256'
        ],
        'rec_dropout_rate': [0.5, 0.3, 0.1],
        'rec_epochs': [5, 10, 15],
    }

    return variables_kg, variables_rec


def define_combined_variables_and_values():
    variables_kg, variables_rec = define_variables_for_KG_and_rec()

    # Combina las variables y posibles valores en un único diccionario para cada tipo
    combined_possible_values = {**variables_kg, **variables_rec}

    return combined_possible_values

def initialize_frequency_for_combined(possible_values):
    # Aquí asumimos que possible_values es un diccionario donde las claves son las variables
    # y los valores son listas de posibles valores para esas variables
    frequency = {var: [1/len(possible_values[var])] * len(possible_values[var]) for var in possible_values}
    return frequency


def setup_model_and_dataloaders_2(solution):
    # Construcción dinámica de las rutas de archivos de embeddings
    kg_model_name = solution['kg_model_name']
    kg_dataset_path = "/home/victor/Escritorio/TFM/git/TFM/Victor/Codigo/GRAFOS/MKR-data/"
    entity_embedding_path = f"{kg_dataset_path}embeddings/{kg_model_name.lower()}/ent_embedding.tsv"
    relation_embedding_path = f"{kg_dataset_path}embeddings/{kg_model_name.lower()}/rel_embedding.tsv"

    # Verifica si existen los archivos de embeddings
    if not os.path.exists(entity_embedding_path) or not os.path.exists(relation_embedding_path):
        print("Uno o más archivos de embeddings no existen. Se continuarán con valores predeterminados.")
        entity_embedding_path = None
        relation_embedding_path = None

    # Extraer hiperparámetros específicos del KG y la recomendación de la solución
    kg_hyperparams = {
        'batch_size': solution['kg_batch_size'],
        'epochs': solution['kg_epochs'],
        'learning_rate': solution['kg_learning_rate'],
        'lmbda': solution['lmbda'],
        'margin': solution['margin'],
        'optimizer': solution['optimizer'],
        'sampling': solution['sampling'],
        'neg_rate': solution['neg_rate'],
        'l1_flag': solution['l1_flag'],
        'alpha': solution['alpha'],
        'label_smoothing': solution['label_smoothing'],
        'hidden_size': solution['embed_dim'],
        'ent_hidden_size': solution['embed_dim'],
        'rel_hidden_size' : solution['embed_dim']
    }



    rec_hyperparams = {
        'batch_size': solution['batch_size'],
        'lr': solution['lr'],
        'embed_dim': solution['embed_dim'],
        'hidden_layers_config': solution['hidden_layers_config'].split('_'),
        'dropout_rate': solution['dropout_rate'],
        'epochs': solution['epochs']
    }

    # Cargar datos y preparar DataLoader
    ratings = np.loadtxt('./MKR-data/ratings_final.txt', dtype=np.int32)
    train_data, eval_data, test_data = dataset_split(ratings)
    train_loader = DataLoader(TrainSet(train_data), batch_size=rec_hyperparams['batch_size'], shuffle=True)
    val_loader = DataLoader(TrainSet(eval_data), batch_size=rec_hyperparams['batch_size'], shuffle=True)
    eval_loader = DataLoader(TrainSet(test_data), batch_size=rec_hyperparams['batch_size'], shuffle=False)

    # Definir dimensiones y número de capas
    _, entity_id2index = read_item_index_to_entity_id_file()
    convert_rating(_)
    entity_id2index, relation_id2index = convert_kg()
    ratings = np.loadtxt('./MKR-data/ratings_final.txt', dtype=np.int32)

    # Parámetros para MultiKR
    user_num = len(np.unique(ratings[:, 0]))  # Número de usuarios únicos
    item_num = len(_)  # Número de ítems únicos
    entity_num = len(entity_id2index)  # Número de entidades únicas
    relation_num = len(relation_id2index)  # Número de relaciones únicas

    n_layer = len(rec_hyperparams['hidden_layers_config'])

    # Instanciar AdvancedMultiKRWithPykg2vec
    model = AdvancedMultiKRWithPykg2vec(
        user_num=user_num,
        item_num=item_num,
        entity_num= entity_num,
        relation_num=relation_num,
        n_layer=n_layer,
        embed_dim=rec_hyperparams['embed_dim'],
        hidden_layers=[int(layer) for layer in rec_hyperparams['hidden_layers_config']],
        dropouts=[rec_hyperparams['dropout_rate']],
        output_rec=1,
        activation_fn='leaky_relu',
        kg_model_name=kg_model_name,
        kg_hyperparams=kg_hyperparams,
        kg_dataset_path=kg_dataset_path,
        kg_trained=False,
        entity_embedding_path=entity_embedding_path,
        relation_embedding_path=relation_embedding_path,
        freeze=False
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=rec_hyperparams['lr'])

    return model, optimizer, train_loader, val_loader, eval_loader


def decode_solution2(solution_array):
    # Aquí definimos el punto de corte basado en la cantidad de variables de KG.
    # Este valor debe ajustarse según la estructura de tus datos.
    cut_point_kg = 13  # Asumiendo 12 variables para KG por defecto

    # Dividir solution_array en partes correspondientes a KG y recomendación
    kg_solution_array = solution_array[:cut_point_kg]
    rec_solution_array = solution_array[cut_point_kg:]

    # Decodifica las partes utilizando funciones específicas
    kg_solution_dict = decode_solution_kg(kg_solution_array)
    rec_solution_dict = decode_solution_rec(rec_solution_array)

    # Combina los diccionarios KG y recomendación
    solution_dict = {**kg_solution_dict, **rec_solution_dict}
    return solution_dict



def decode_solution_kg(kg_solution_array):
    print("Decodificando variables KG...")
    kg_solution_dict = {
        'kg_batch_size': kg_solution_array[0],
        'kg_epochs': kg_solution_array[1],
        'kg_learning_rate': kg_solution_array[2],
        'kg_model_name': kg_solution_array[3],
        'lmbda': kg_solution_array[4],
        'margin': kg_solution_array[5],
        'optimizer': kg_solution_array[6],
        'sampling': kg_solution_array[7],
        'neg_rate': kg_solution_array[8],
        'rel_hidden_size': kg_solution_array[9],
        'l1_flag': kg_solution_array[10],
        'alpha': kg_solution_array[11],
        'label_smoothing': kg_solution_array[12],
    }
    return kg_solution_dict


def decode_solution_rec(solution_array):
    # Ajuste a la nueva estructura y orden de variables de recomendación
    return {
        'batch_size': int(solution_array[0]),
        'lr': solution_array[1],
        'embed_dim': int(solution_array[2]),
        'hidden_layers_config': solution_array[3],
        'dropout_rate': solution_array[4],
        'epochs': int(solution_array[5])  # Se elimina 'output_rec' y se ajusta el índice para 'epochs'
    }



def multiKR_cost_function_eda(solution):
    """
    Función de costo que integra tanto el entrenamiento del KG como el del sistema de recomendación.
    """
    # Configurar el modelo y dataloaders
    model, optimizer, train_loader, val_loader, eval_loader = setup_model_and_dataloaders_2(solution)

    # Entrenar el modelo KG primero
    if not model.kg_trained:
        model.train_kg_model(in_notebook=False)





    loss_function = torch.nn.BCEWithLogitsLoss()

    epochs = solution['epochs']  # Extrae epochs del diccionario de solución

    # Entrenar el modelo
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, loss_function, epoch, "rec")

    # Evaluar el modelo usando val_loader para no filtrar información del conjunto de test
    auc = evaluate_model(model, val_loader, epochs - 1)

    # Devolver el negativo de AUC para minimización
    return -auc




def multiKR_cost_wrapper_eda(solution_array):
    # Decodifica directamente la solución a partir de solution_array
    solution_dict = decode_solution2(solution_array)
    print(f"Los hiperparámetros seleccionados son: {solution_dict}")
    # Llama a la función de costo con el diccionario decodificado
    return multiKR_cost_function_eda(solution_dict)