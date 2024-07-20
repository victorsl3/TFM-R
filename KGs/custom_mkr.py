import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


def linear_layer(input_dim, output_dim, dropout_rate=0.0, activation='leaky_relu'):
    """
    Crea una capa lineal con una función de activación personalizable y dropout.
    
    Parámetros:
    - input_dim: Dimensiones de entrada.
    - output_dim: Dimensiones de salida.
    - dropout_rate: Tasa de dropout (por defecto, 0.0).
    - activation: Tipo de función de activación. Soporta 'relu', 'leaky_relu', 'tanh', 'sigmoid', etc.
    
    Retorna:
    - Una secuencia de operaciones (nn.Sequential) que incluye la capa lineal, la activación y el dropout.
    """
    layers = [nn.Linear(input_dim, output_dim)]
    
    # Selecciona la función de activación
    if activation == 'relu':
        layers.append(nn.ReLU())
    elif activation == 'leaky_relu':
        layers.append(nn.LeakyReLU())
    elif activation == 'tanh':
        layers.append(nn.Tanh())
    elif activation == 'sigmoid':
        layers.append(nn.Sigmoid())

    
    # Agrega dropout si es especificado
    if dropout_rate > 0:
        layers.append(nn.Dropout(dropout_rate))
    
    return nn.Sequential(*layers)

class MultiKR(nn.Module):
    """
    Modelo MultiKR para recomendaciones mejoradas por grafos de conocimiento (KG).
    """
    def __init__(self, user_num, item_num, entity_num, relation_num, n_layer, embed_dim, hidden_layers, dropouts, output_rec, activation_fn='leaky_relu'):
        super(MultiKR, self).__init__()

        self.n_layer = n_layer  # Guardar n_layer como un atributo de la instancia
        self.activation_fn = activation_fn  # Nueva propiedad para almacenar la función de activación preferida

        # Inicialización de embeddings
        self.user_embed = nn.Embedding(user_num, embed_dim)
        self.item_embed = nn.Embedding(item_num, embed_dim)
        self.entity_embed = nn.Embedding(entity_num, embed_dim)
        self.relation_embed = nn.Embedding(relation_num, embed_dim)

        # Unidades de compresión cruzada dinámicas
        self.init_cross_compression_units(embed_dim)

        # MLPs para submodelos de KG y recomendación
        self.kg_layers = self._build_mlp_layers(2*embed_dim, hidden_layers, dropouts, embed_dim)
        self.rec_layers = self._build_mlp_layers(2*embed_dim, hidden_layers, dropouts, output_rec)

        # Definir una capa MLP simple para el procesamiento de embeddings de usuario
        self.user_low_mlp_layer = linear_layer(embed_dim, embed_dim, dropout_rate=0.5, activation='leaky_relu')


        self._init_weights()



    def init_cross_compression_units(self, embed_dim):
        """
        Inicializa los pesos y sesgos para la unidad de compresión cruzada.
        """
        self.compress_weights = nn.ParameterDict({
            'vv': nn.Parameter(torch.rand(embed_dim, 1)),
            'ev': nn.Parameter(torch.rand(embed_dim, 1)),
            've': nn.Parameter(torch.rand(embed_dim, 1)),
            'ee': nn.Parameter(torch.rand(embed_dim, 1))
        })
        self.compress_biases = nn.ParameterDict({
            'v': nn.Parameter(torch.rand(1)),
            'e': nn.Parameter(torch.rand(1))
        })

    def _build_mlp_layers(self, input_dim, hidden_layers, dropouts, output_dim):
        """
        Construye secuencialmente las capas MLP con la función de activación seleccionada.
        """
        mlp_layers = nn.Sequential()
        layers = [input_dim] + hidden_layers + [output_dim]
        for i in range(len(layers) - 1):
            mlp_layers.add_module(f'layer_{i}', linear_layer(layers[i], layers[i+1], dropout_rate=dropouts[i] if i < len(dropouts) else 0, activation=self.activation_fn))
        return mlp_layers

    def _init_weights(self):
        """
        Inicializa los pesos de los embeddings y las unidades de compresión cruzada.
        """
        nn.init.normal_(self.user_embed.weight, std=0.01)
        nn.init.normal_(self.item_embed.weight, std=0.01)
        for param in self.compress_weights.values():
            nn.init.xavier_uniform_(param)
        for bias in self.compress_biases.values():
            nn.init.zeros_(bias)


    def cross_compress_unit(self, item_embed, head_embed):
        """
        Unidad de Compresión Cruzada para interacciones entre ítems y entidades.
        Realiza una operación de compresión cruzada y devuelve embeddings comprimidos.

        Parámetros:
        - item_embed: Embedding del ítem, dimensión [batch_size, embed_dim].
        - head_embed: Embedding de la entidad, dimensión [batch_size, embed_dim].

        Retorna:
        - item_embed_c: Embedding del ítem comprimido.
        - head_embed_c: Embedding de la entidad comprimido.
        """
        item_embed_reshape = item_embed.unsqueeze(-1)
        head_embed_reshape = head_embed.unsqueeze(-1)
        c = torch.matmul(item_embed_reshape, head_embed_reshape.transpose(1, 2))
        c_t = c.transpose(1, 2)
        
        # Usa las claves correctas del nn.ParameterDict para acceder a los pesos y sesgos
        item_embed_c = torch.matmul(c, self.compress_weights['vv']) + torch.matmul(c_t, self.compress_weights['ev']) + self.compress_biases['v']
        head_embed_c = torch.matmul(c, self.compress_weights['ve']) + torch.matmul(c_t, self.compress_weights['ee']) + self.compress_biases['e']

        return item_embed_c.squeeze(), head_embed_c.squeeze()


    def forward(self, data, train_type):
        """
        Define la pasada hacia adelante del modelo.

        Parámetros:
        - data: Datos de entrada.
        - train_type: Tipo de entrenamiento ('rec' para recomendación, 'kg' para KG).

        Retorna:
        - Salida del modelo según el tipo de entrenamiento.
        """
        try:
            if train_type == 'rec':
                user_embed, item_embed, head_embed, rec_target = self._prepare_embeddings(data)
                for _ in range(self.n_layer):
                    user_embed = self.user_low_mlp_layer(user_embed)
                    item_embed, head_embed = self.cross_compress_unit(item_embed, head_embed)
                high_layer = torch.cat((user_embed, item_embed), dim=1)
                rec_out = self.rec_layers(high_layer)
                return rec_out.squeeze(), rec_target
            else:
                head_embed, item_embed, relation_embed, tail_embed = self._prepare_embeddings_kg(data)
                for _ in range(self.n_layer):
                    item_embed, head_embed = self.cross_compress_unit(item_embed, head_embed)
                    relation_embed = self.relation_low_mlp_layer(relation_embed)
                high_layer = torch.cat((head_embed, relation_embed), dim=1)
                tail_out = self.kg_layers(high_layer)
                return tail_out, tail_embed
        except IndexError as e:
            print("IndexError caught in forward pass:", e)
            print("Data received:", data)
            raise


    # def _prepare_embeddings(self, data):
    #     """Prepara los embeddings para el submodelo de recomendación."""
    #     user_embed = self.user_embed(data[0].long())
    #     item_embed = self.item_embed(data[1].long())
    #     head_embed = self.entity_embed(data[1].long())
    #     rec_target = data[2].float()
    #     return user_embed, item_embed, head_embed, rec_target

    # def _prepare_embeddings_kg(self, data):
    #     """Prepara los embeddings para el submodelo de KG."""
    #     head_embed = self.entity_embed(data[0].long())
    #     item_embed = self.item_embed(data[0].long())
    #     relation_embed = self.relation_embed(data[1].long())
    #     tail_embed = self.entity_embed(data[2].long())
    #     return head_embed, item_embed, relation_embed, tail_embed

    def _prepare_embeddings(self, data):
        user_ids, item_ids = data[0].long(), data[1].long()
        # print("Max user id:", user_ids.max().item(), "Max item id:", item_ids.max().item())  # Verificar los índices máximos
        user_embed = self.user_embed(user_ids)
        item_embed = self.item_embed(item_ids)
        head_embed = self.entity_embed(item_ids)  # Asegúrate de que este es el uso correcto de item_ids para head_embed
        rec_target = data[2].float()
        return user_embed, item_embed, head_embed, rec_target

    def _prepare_embeddings_kg(self, data):
        head_ids = data[0].long()
        item_ids = data[1].long()  # Verifica si debería ser data[1] o otro índice
        relation_ids = data[2].long()
        tail_ids = data[3].long()
        # print("KG embedding access - Head max:", head_ids.max().item(), "Item max:", item_ids.max().item())
        head_embed = self.entity_embed(head_ids)
        item_embed = self.item_embed(item_ids)
        relation_embed = self.relation_embed(relation_ids)
        tail_embed = self.entity_embed(tail_ids)
        return head_embed, item_embed, relation_embed, tail_embed
