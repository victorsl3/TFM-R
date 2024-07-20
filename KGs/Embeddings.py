import torch
import torch.nn.functional as F

# hu (head embedding): El embedding de la entidad de cabeza en la tripleta.
# hr (relation embedding): El embedding de la relación en la tripleta.
# hv (tail embedding): El embedding de la entidad de cola en la tripleta.
# hu1, hu2, hv1, hv2 (auxiliary entity embeddings): En algunos modelos como TransD, estos representan embeddings auxiliares o proyecciones específicas 
#   de las entidades de cabeza y cola, respectivamente, utilizadas para ajustar los embeddings en función de la relación.
# hr1, hr2 (auxiliary relation embeddings): En modelos como TransH y TransD, estos representan embeddings auxiliares o 
#   proyecciones específicas de la relación, utilizadas para ajustar los embeddings de las entidades de cabeza y cola en 
#   función de la naturaleza específica de la relación.


import torch

class TransE:
    """
    Modelo TransE para embeddings de grafos de conocimiento. Calcula la distancia (norma L2)
    entre la suma de los embeddings de la entidad de cabeza y la relación, y el embedding de la entidad de cola.
    """
    def score(self, hu, hr, hv):
        return -torch.norm(hu + hr - hv, p=2, dim=-1)

class DistMult:
    """
    Modelo DistMult para embeddings de grafos de conocimiento. Calcula el score de una tripleta
    realizando un producto elemento a elemento entre los embeddings de la entidad de cabeza,
    la relación, y la entidad de cola, y sumando los resultados.
    """
    def score(self, hu, hr, hv):
        return torch.sum(hu * hr * hv, dim=-1)

class TransH:
    """
    Modelo TransH para embeddings de grafos de conocimiento. Proyecta los embeddings de la entidad de cabeza
    y cola en un hiperplano definido por la relación antes de calcular la distancia.
    """
    def score(self, hu, hr, hv, wr, dr):
        # Normalizar wr para asegurar que sea un vector normal al hiperplano
        wr_normalized = F.normalize(wr, p=2, dim=-1)
        # Proyectar hu y hv sobre el hiperplano
        hu_proj = hu - torch.sum(hu * wr_normalized, dim=-1, keepdim=True) * wr_normalized
        hv_proj = hv - torch.sum(hv * wr_normalized, dim=-1, keepdim=True) * wr_normalized
        # Calcular la puntuación utilizando el vector de traducción dr
        return -torch.norm(hu_proj + dr - hv_proj, p=2, dim=-1)


class TransD:
    """
    Modelo TransD para embeddings de grafos de conocimiento. Aplica transformaciones específicas de la relación
    a los embeddings de la entidad de cabeza y cola antes de calcular la distancia.
    """
    def score(self, hu, hr, hv, hp, rp):
        # Crear matrices de mapeo dinámico para la cabeza y la cola
        M_rh = torch.outer(hp, rp) + torch.eye(hp.size(0))
        M_rt = torch.outer(hp, rp) + torch.eye(hp.size(0))
        
        # Proyectar hu y hv al espacio de la relación
        hu_proj = torch.matmul(M_rh, hu)
        hv_proj = torch.matmul(M_rt, hv)
        
        # Calcular la puntuación como la norma L2 de la diferencia entre hu_proj + hr y hv_proj
        return -torch.norm(hu_proj + hr - hv_proj, p=2, dim=-1)
    


class KnowledgeGraphEmbeddings:
    def __init__(self, num_entities, num_relations, entity_dim, relation_dim):
        # Inicializar embeddings auxiliares para TransH
        self.wr = torch.nn.ParameterDict({str(r): torch.nn.Parameter(torch.randn(entity_dim))
                                          for r in range(num_relations)})
        self.dr = torch.nn.ParameterDict({str(r): torch.nn.Parameter(torch.randn(entity_dim))
                                          for r in range(num_relations)})
        
        # Inicializar embeddings auxiliares para TransD
        self.hp = torch.nn.ParameterDict({str(e): torch.nn.Parameter(torch.randn(entity_dim))
                                          for e in range(num_entities)})
        self.rp = torch.nn.ParameterDict({str(r): torch.nn.Parameter(torch.randn(relation_dim))
                                          for r in range(num_relations)})



class RotatE:
    """
    Modelo RotatE para embeddings de grafos de conocimiento. Realiza una rotación en el espacio complejo
    de los embeddings de la entidad de cabeza por el embedding de la relación antes de calcular la distancia
    hasta el embedding de la entidad de cola.
    """
    def score(self, hu, hr, hv):
        hr_mag = torch.sqrt(hr[..., 0]**2 + hr[..., 1]**2) + 1e-9
        hr = hr / hr_mag.unsqueeze(-1)
        hu_hr = hu * hr
        return -torch.norm(hu_hr - hv, p=2, dim=-1)

class QuatE:
    """
    Modelo QuatE para embeddings de grafos de conocimiento. Utiliza cuaterniones para representar
    las entidades y las relaciones, realizando un producto Hamilton antes de calcular el producto punto.
    """
    def score(self, hu, hr, hv):
        def hamilton_product(q, r):
            t0 = q[..., 0] * r[..., 0] - q[..., 1] * r[..., 1] - q[..., 2] * r[..., 2] - q[..., 3] * r[..., 3]
            t1 = q[..., 0] * r[..., 1] + q[..., 1] * r[..., 0] + q[..., 2] * r[..., 3] - q[..., 3] * r[..., 2]
            t2 = q[..., 0] * r[..., 2] - q[..., 1] * r[..., 3] + q[..., 2] * r[..., 0] + q[..., 3] * r[..., 1]
            t3 = q[..., 0] * r[..., 3] + q[..., 1] * r[..., 2] - q[..., 2] * r[..., 1] + q[..., 3] * r[..., 0]
            return torch.stack((t0, t1, t2, t3), dim=-1)
        
        hu_hr = hamilton_product(hu, hr)
        return torch.sum(hu_hr * hv, dim=-1)