from custom_mkr import MultiKR, linear_layer
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from poincareball import PoincareBall
from hyperbolic.poincare import Point
from Embeddings import *

class CustomGCN(nn.Module):
    """
    Esta clase inicializa una red neuronal convolucional para grafos (GCN),
    con la capacidad opcional de realizar operaciones en el espacio hiperbólico.
    
    Parámetros:
    - in_channels (int): Número de características de entrada por nodo.
    - hidden_channels (list[int]): Lista con el número de características en cada capa oculta.
    - out_channels (int): Número de características de salida para los nodos.
    - num_layers (int): Número total de capas en la GCN.
    - dropout (float): Tasa de dropout aplicada a las características de los nodos.
    - use_attention (bool): Si es True, usa capas GATConv para atención.
    - use_skip_connections (bool): Si es True, añade conexiones residuales entre las capas.
    - activation_fn (str): Función de activación usada después de las capas lineales.
    - use_hyperbolic (bool): Indica si se deben realizar transformaciones hiperbólicas.
    - poincare_radius (float): Radio de la bola de Poincaré.

    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_attention=False, use_skip_connections=True, activation_fn='leaky_relu', use_hyperbolic=False, poincare_radius=1, embedding_type='TransE'):
        super(CustomGCN, self).__init__()      
        self.layers = nn.ModuleList()
        self.use_skip_connections = use_skip_connections
        self.dropout = dropout
        self.use_hyperbolic = use_hyperbolic
        self.poincare_ball = PoincareBall(radius=poincare_radius) if use_hyperbolic else None
        self.embedding_type = embedding_type  # Tipo de embedding
        self.embedding_model = self._select_embedding_model()  # Instancia del modelo de embedding


        # Inicializar las capas del GCN, ya sean GCNConv o GATConv dependiendo de use_attention
        channels = [in_channels] + hidden_channels + [out_channels]
        for i in range(1, len(channels)):
            in_c, out_c = channels[i-1], channels[i]
            conv_layer = geom_nn.GATConv(in_c, out_c) if use_attention else geom_nn.GCNConv(in_c, out_c, add_self_loops=True)
            self.layers.append(conv_layer)
            ff_layer = linear_layer(out_c, out_c, dropout_rate=dropout, activation=activation_fn)
            self.layers.append(ff_layer)
            if use_skip_connections and i > 1:
                self.layers.append(nn.Linear(channels[i-2], out_c))
            self.layers.append(nn.BatchNorm1d(out_c))
    

    def _select_embedding_model(self):
        # Diccionario que mapea nombres de tipos de embedding a sus respectivas clases
        embedding_models = {
            'TransE': TransE,
            'DistMult': DistMult,
            'TransH': TransH,
            'TransD': TransD,
            'RotatE': RotatE,
            'QuatE': QuatE
        }
        # Retorna una instancia del modelo de embedding seleccionado
        return embedding_models.get(self.embedding_type, TransE)()  # Usa TransE como default si no se encuentra el tipo
    
    def apply_embedding_model(self, hu, hr, hv):
        """
        Aplica el modelo de embeddings seleccionado a las tripletas.
        hu, hr, hv son los embeddings de las entidades de cabeza, relación y cola, respectivamente.
        """
        if self.embedding_model:
            hu_embedded, hr_embedded, hv_embedded = self.embedding_model(hu, hr, hv)
        else:
            # Si no se proporciona un modelo de embeddings, usar las entradas directamente
            hu_embedded, hr_embedded, hv_embedded = hu, hr, hv
        return hu_embedded, hr_embedded, hv_embedded

    def forward(self, x, edge_index):
        """
        Propaga los datos a través de la red GCN. 
        Si se activa el uso de transformaciones hiperbólicas, transforma los datos al espacio hiperbólico antes y después de la propagación.
        
        - x (Tensor): Tensor con las características de los nodos.
        - edge_index (Tensor): Tensor que define las conexiones entre nodos (aristas).
        
        Retorna:
        - Tensor con las características de los nodos después de la propagación por la GCN.
        """
        # if self.use_hyperbolic:
        #     # Transformar 'x' al espacio hiperbólico si se requiere
        #     x = self.exp_map_batch(x)
        
        # Propagación estándar a través de las capas de GCN
        x_skip = x
        for layer in self.layers:
            if isinstance(layer, (geom_nn.GCNConv, geom_nn.GATConv)):
                x = layer(x, edge_index)
            elif isinstance(layer, nn.Sequential):
                x = layer(x)
            elif isinstance(layer, nn.Linear):
                x_skip = layer(x_skip)
            elif isinstance(layer, nn.BatchNorm1d):
                x = layer(x)
            
            if self.use_skip_connections:
                x += x_skip
                x = F.relu(x)
                x_skip = x
            
            x = F.dropout(x, p=self.dropout, training=self.training)

        # if self.use_hyperbolic:
        #     # Transformar 'x' de vuelta al espacio latente
        #     x = self.log_map_batch(x)
        
        return x

    # def exp_map_batch(self, tangent_vectors):
    #     """Aplica el mapa exponencial a un lote de vectores tangentes."""
    #     return torch.stack([self.poincare_ball.exp_map(Point(0, 0), Point(v[0], v[1])) for v in tangent_vectors])

    # def log_map_batch(self, points):
    #     """Aplica el mapa logarítmico a un lote de puntos en la bola de Poincaré."""
    #     return torch.stack([self.poincare_ball.log_map(Point(0, 0), p) for p in points])



class MultiKRWithGCN(MultiKR):
    def __init__(self, user_num, item_num, entity_num, relation_num, n_layer, embed_dim, hidden_layers, dropouts, output_rec, gcn_params):
        super(MultiKRWithGCN, self).__init__(user_num, item_num, entity_num, relation_num, n_layer, embed_dim, hidden_layers, dropouts, output_rec, 'leaky_relu')
        self.gcn = CustomGCN(embed_dim, gcn_params['hidden_channels'], gcn_params['out_channels'], gcn_params['num_layers'], gcn_params['dropout'], gcn_params['use_attention'], gcn_params['use_skip_connections'])
        adjusted_dim = embed_dim + gcn_params['out_channels'] if gcn_params['use_skip_connections'] else gcn_params['out_channels']
        self.kg_layers = self._build_mlp_layers(adjusted_dim, hidden_layers, dropouts, embed_dim)

    def forward(self, data, train_type, edge_index=None, relation_index=None):
        print(f"Forward call - Task Type: {train_type}, Edge Index Provided: {edge_index is not None}, Relation Index Provided: {relation_index is not None}")
        if train_type == 'kg':
            if edge_index is None or relation_index is None:
                print("Error: edge_index and relation_index must be provided for 'kg' task")
                return
            print(f"edge_index size: {edge_index.size()}, relation_index size: {relation_index.size()}")

            entity_embeddings = self.entity_embed.weight
            print(f"Entity Embeddings Size: {entity_embeddings.size()}")
            # Pass both edge_index and relation_index to the GCN
            gcn_entity_embeddings = self.gcn(entity_embeddings, edge_index, relation_index=relation_index)

            max_head_idx, max_relation_type_idx, max_tail_idx = data[0].max().item(), relation_index.max().item(), data[2].max().item()
            print(f"Data - Max head index: {max_head_idx}, Max relation type index: {max_relation_type_idx}, Max tail index: {max_tail_idx}")

            if max_head_idx >= self.entity_embed.num_embeddings or max_tail_idx >= self.entity_embed.num_embeddings:
                print("Warning: Entity index out of range.")
            if max_relation_type_idx >= self.relation_embed.num_embeddings:
                print("Warning: Relation type index out of range.")

            head_embed = gcn_entity_embeddings[data[0].long()]
            relation_embed = self.relation_embed(relation_index.long())  # Correctly use relation_index here
            tail_embed = gcn_entity_embeddings[data[2].long()]

            for _ in range(self.n_layer):
                head_embed, tail_embed = self.cross_compress_unit(head_embed, tail_embed)

            kg_high_layer = torch.cat((head_embed, relation_embed), dim=1)
            kg_output = self.kg_layers(kg_high_layer)

            return kg_output, tail_embed
        else:
            return super(MultiKRWithGCN, self).forward(data, train_type)



