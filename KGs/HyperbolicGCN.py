import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as geom_nn
from poincareball import PoincareBall
from GNN import CustomGCN

class HyperbolicGCN(CustomGCN):
    """
    Clase para una GCN que opera en el espacio hiperbólico.
    
    Asume que 'CustomGCN' es una clase base definida previamente
    que implementa una red convolucional para grafos (Graph Convolutional Network).
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, use_attention=False, use_skip_connections=True, activation_fn='leaky_relu', poincare_radius=1):
        super(HyperbolicGCN, self).__init__(in_channels, hidden_channels, out_channels, num_layers, dropout, use_attention, use_skip_connections, activation_fn)
        self.poincare_ball = PoincareBall(radius=poincare_radius)
    
    def exp_map_layer(self, tangent_vectors):
        """Aplica el mapa exponencial a un conjunto de vectores tangentes."""
        base_point = self.poincare_ball.radius  # O el punto base que se desee
        return [self.poincare_ball.exp_map(base_point, v) for v in tangent_vectors]
    
    def log_map_layer(self, points):
        """Aplica el mapa logarítmico a un conjunto de puntos en la bola de Poincaré."""
        base_point = self.poincare_ball.radius  # O el punto base que se desee
        return [self.poincare_ball.log_map(base_point, p) for p in points]
    
    def mobius_addition_layer(self, points_x, points_y):
        """Aplica la adición de Möbius a pares de puntos."""
        return [self.poincare_ball.mobius_addition(x, y) for x, y in zip(points_x, points_y)]
    
    def forward(self, x, edge_index):
        """
        Propaga los datos a través de la red GCN en el espacio hiperbólico.

        Parámetros:
        - x (Tensor): Tensor de características de los nodos.
        - edge_index (Tensor): Tensor que define las conexiones entre nodos (aristas).

        Retorna:
        - Tensor: Características de los nodos después de pasar por la GCN.
        """
        # Supongamos que 'x' ya está en la forma de vectores tangentes adecuados para el espacio hiperbólico
        x = self.exp_map_layer(x)  # Mapeo exponencial para llevar los vectores al espacio hiperbólico
        
        # Propagación a través de las capas de GCN como se definiría normalmente
        x = super(HyperbolicGCN, self).forward(x, edge_index)
        
        # Aplicación del mapa logarítmico para volver al espacio tangente
        x = self.log_map_layer(x)
        
        return x
