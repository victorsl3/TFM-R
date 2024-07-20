import torch
import torch.nn as nn
from pykg2vec.common import Importer, KGEArgParser
from pykg2vec.utils.trainer import Trainer
from pykg2vec.data.kgcontroller import KnowledgeGraph
from custom_mkr import MultiKR, linear_layer
import glob
import os

class AdvancedMultiKR(MultiKR):
    def __init__(self, user_num, item_num, entity_num, relation_num, n_layer, embed_dim, hidden_layers, dropouts, output_rec, activation_fn='leaky_relu', kg_model_name='transe', kg_hyperparams=None):
        super(AdvancedMultiKR, self).__init__(user_num, item_num, entity_num, relation_num, n_layer, embed_dim, hidden_layers, dropouts, output_rec, activation_fn)
        
        self.kg_model_name = kg_model_name
        self.kg_hyperparams = kg_hyperparams or {}
        
        # Cargar la configuración y el modelo de pykg2vec para KG
        self.config, self.kg_model = self.load_pykg2vec_model(kg_model_name, self.kg_hyperparams)
        
        # Preparación adicional si es necesario
        self.trainer = Trainer(model=self.kg_model, config=self.config)
        self.trainer.build_model()
        
        # Inicializa el controlador del grafo de conocimiento
        self.knowledge_graph = KnowledgeGraph(dataset=self.config.dataset_name, custom_dataset_path=self.config.dataset_path)

        # Asegura que el modelo y el optimizador estén configurados para el dispositivo adecuado
        self.kg_model.to(self.config.device)
        self.optimizer = self.trainer.optimizer

        # Asegúrate de que la cantidad de embeddings de relación coincida con el número de relaciones únicas
        self.relation_embed = nn.Embedding(relation_num, embed_dim)  # Ajusta relation_num al número de relaciones únicas



    def load_pykg2vec_model(self, model_name, hyperparams):
        """Carga la configuración y el modelo especificado de pykg2vec con hiperparámetros personalizados."""
        args_list = ['-mn', model_name]
        args = KGEArgParser().get_args(args_list)
        config_def, model_def = Importer().import_model_config(model_name)
        config = config_def(args)

        for key, value in hyperparams.items():
            if hasattr(config, key):
                setattr(config, key, value)
            else:
                print(f"Advertencia: El hiperparámetro '{key}' no es reconocido por el modelo '{model_name}' y será ignorado.")

        # Supongamos que el modelo espera parámetros específicos durante la inicialización
        # y no un objeto de configuración. Aquí necesitas adaptar la creación del modelo
        # según sus requisitos específicos. Este es un ejemplo genérico:
        try:
            # Si model_def espera un objeto de configuración directamente
            model = model_def(config)
        except TypeError:
            # Si model_def no espera un objeto de configuración, ajusta según los parámetros requeridos
            print("model_def no espera un objeto de configuración, ajusta según los parámetros requeridos")
            model = model_def(**config.__dict__)

        return config, model



    # def forward(self, data, train_type):
    #     if train_type == 'kg':
    #         # Cambio aquí: Aceptar cuatro elementos en vez de tres
    #         head_idx, relation_idx, tail_idx, label = data

    #         # 1. Obtener embeddings de TransE
    #         head_embed = self.kg_model.ent_embeddings(head_idx)
    #         relation_embed = self.kg_model.rel_embeddings(relation_idx)
    #         tail_embed = self.kg_model.ent_embeddings(tail_idx)

    #         # 2. Combinar embeddings (aquí se muestra una concatenación como ejemplo)
    #         combined_embed = torch.cat([head_embed, relation_embed, tail_embed], dim=-1)

    #         # 3. Pasar la representación combinada a través de 'kg_layers' (MLP)
    #         kg_output = self.kg_layers(combined_embed)

    #         # Retorna la salida del MLP para la parte KG
    #         return kg_output
    #     else:
    #         # Para la recomendación, se espera que los datos ya estén en el formato adecuado
    #         # Asumiendo que 'data' es una tupla de (users, items) para recomendaciones
    #         user_idx, item_idx = data[0], data[1]  # Desempaqueta usuarios e ítems

    #         # Se puede añadir un paso adicional aquí si es necesario para preparar
    #         # los embeddings o cualquier otra pre-procesamiento específico para recomendaciones

    #         # Llama a super().forward pasando los datos en el formato esperado
    #         return super().forward((user_idx, item_idx), train_type)



    def forward(self, data, train_type):
        if train_type == 'kg':
            head_idx, relation_idx, tail_idx = data  # No esperes la etiqueta aquí

            head_embed = self.kg_model.ent_embeddings(head_idx)
            relation_embed = self.kg_model.rel_embeddings(relation_idx)
            tail_embed = self.kg_model.ent_embeddings(tail_idx)

            combined_embed = torch.cat([head_embed, relation_embed, tail_embed], dim=-1)
            kg_output = self.kg_layers(combined_embed)

            return kg_output
        else:
            return super().forward(data, train_type)


#######################################################################################################################################################################################################


import tempfile
import yaml
import subprocess
import os
from custom_mkr import MultiKR, linear_layer
from IPython.display import clear_output
import time

def load_embeddings_from_tsv(file_path, embed_dim):
    embeddings = []
    with open(file_path, 'r') as file:
        for line in file:
            values = line.strip().split('\t')
            vector = [float(v) for v in values[:embed_dim]]  # Asegura que cada vector tenga la dimensión correcta
            embeddings.append(vector)
    return torch.tensor(embeddings, dtype=torch.float)

def load_embeddings_and_ids(embedding_path, label_path, embed_dim):
    # Cargar embeddings
    embeddings = load_embeddings_from_tsv(embedding_path, embed_dim)
    
    # Cargar IDs
    id_to_index = {}
    with open(label_path, 'r') as file:
        for index, line in enumerate(file):
            id_str = line.strip()
            id_to_index[id_str] = index
    
    return embeddings, id_to_index

class AdvancedMultiKRWithPykg2vec(MultiKR):
    def __init__(self, user_num, entity_num, relation_num, item_num, n_layer, embed_dim, hidden_layers, dropouts, output_rec, activation_fn, kg_model_name, kg_hyperparams, kg_dataset_path, kg_trained=False, entity_embedding_path=None, relation_embedding_path=None, freeze=False):
        super(AdvancedMultiKRWithPykg2vec, self).__init__(user_num, item_num, entity_num, relation_num, n_layer, embed_dim, hidden_layers, dropouts, output_rec, activation_fn)

        self.kg_dataset_path = kg_dataset_path
        self.kg_model_name = kg_model_name
        self.kg_hyperparams = kg_hyperparams
        self.freeze = freeze
        self.embed_dim = embed_dim
        self.kg_trained=kg_trained

        # Carga los embeddings solo si el KG ha sido entrenado y se proporcionan las rutas
        if kg_trained and entity_embedding_path and relation_embedding_path:
            entity_label_path = entity_embedding_path.replace("ent_embedding.tsv", "ent_labels.tsv")
            relation_label_path = relation_embedding_path.replace("rel_embedding.tsv", "rel_labels.tsv")
            
            # Incluir el parámetro embed_dim en la llamada a set_embeddings
            self.set_embeddings(entity_embedding_path, entity_label_path, relation_embedding_path, relation_label_path, self.embed_dim, freeze)
        else:
            print("KG model not trained or embedding paths not provided. Skipping embedding initialization.")



    def train_kg_model(self, in_notebook=True):
        # Crea un archivo temporal para los hiperparámetros
        with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.yaml') as tmpfile:
            hyperparams_config = {
                'model_name': self.kg_model_name,
                'datasets': [{
                    'dataset': 'custom_dataset',
                    'parameters': self.kg_hyperparams
                }]
            }
            yaml.dump(hyperparams_config, tmpfile)
            tmpfile_path = tmpfile.name

        # Actualiza el comando para usar el archivo de hiperparámetros
        command = f"pykg2vec-train -exp True -mn {self.kg_model_name} -ds custom_dataset -dsp '{self.kg_dataset_path}' -hpf '{tmpfile_path}'"

        if in_notebook:
            process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True)
            while True:
                output = process.stdout.readline()
                if output == '' and process.poll() is not None:
                    break
                if output:
                    print(output.strip())
        else:
            # Ejecuta el comando sin capturar la salida para ejecución fuera de notebooks
            process=subprocess.run(command, shell=True)

        if process.returncode == 0:
            self.kg_trained=True
            print("Entrenamiento completado con éxito.")

        # Usando la función find_embedding_file
        base_path = f"/home/victor/Escritorio/TFM/git/TFM/Victor/Codigo/GRAFOS/MKR-data/embeddings/{self.kg_model_name.lower()}"

        try:
            self.entity_embedding_path = find_embedding_file(base_path, 'ent')
            self.entity_label_path = os.path.join(base_path, "ent_labels.tsv")
            self.relation_embedding_path = find_embedding_file(base_path, 'rel')
            self.relation_label_path = os.path.join(base_path, "rel_labels.tsv")
        except FileNotFoundError as e:
            print(e)
            print("Error en el entrenamiento del modelo.")



        # Borra el archivo temporal de hiperparámetros
        os.remove(tmpfile_path)


    def set_embeddings(self, entity_embedding_path, entity_label_path, relation_embedding_path, relation_label_path, embed_dim, freeze=False):
        entity_embeddings, self.entity_id_to_index = load_embeddings_and_ids(entity_embedding_path, entity_label_path, embed_dim)
        relation_embeddings, self.relation_id_to_index = load_embeddings_and_ids(relation_embedding_path, relation_label_path, embed_dim)
        
        self.entity_embed = nn.Embedding.from_pretrained(entity_embeddings, freeze=freeze)
        self.relation_embed = nn.Embedding.from_pretrained(relation_embeddings, freeze=freeze)
        print("Embeddings cargados")



import glob
import os


def find_embedding_file(base_path, file_type, entity_special_cases=None):
    """
    Busca archivos de embeddings siguiendo una jerarquía de patrones de nombres.
    
    :param base_path: Ruta base donde se buscarán los archivos.
    :param file_type: Tipo de archivo a buscar ('ent' o 'rel').
    :param entity_special_cases: Casos especiales para el manejo de archivos de entidades.
    :return: Ruta al archivo encontrado siguiendo la jerarquía especificada.
    """
    # Definir los patrones de búsqueda jerárquicos para entidades y relaciones
    if file_type == 'ent':
        entity_special_cases = entity_special_cases or ['embedding', 'embeddings', 'obj_embedding', 'embeddings_mu', 'head_embedding']
        patterns = [f"{file_type}_{case}.tsv" for case in entity_special_cases]
        # Añadir un caso especial para 'obj'
        if 'obj_embedding' in entity_special_cases:
            patterns.append("obj_embedding.tsv")
    else:  # file_type == 'rel'
        patterns = [
            "rel_embedding.tsv",
            "rel_embeddings.tsv",
            "rel_embeddings_mu.tsv",
            "rel_head_embedding.tsv"
        ]

    # Intentar encontrar el archivo siguiendo la jerarquía de patrones
    for pattern in patterns:
        full_path = os.path.join(base_path, pattern)
        if os.path.exists(full_path):
            return full_path

    raise FileNotFoundError(f"No se encontraron archivos de embedding para '{file_type}' en '{base_path}' siguiendo la jerarquía especificada.")
