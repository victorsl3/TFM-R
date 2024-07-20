from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from custom_mkr import MultiKR
from tqdm import tqdm
from Embeddings import *


def train_and_evaluate(model, train_loader, val_loader, optimizer, loss_function, num_epochs, task_type, edge_index=None, relation_index=None):
    for epoch in range(num_epochs):
        model.train()  # Poner el modelo en modo entrenamiento
        train_loss = 0.0
        train_predictions, train_targets = [], []  # Listas para recolectar predicciones y objetivos para el AUC

        # Envolver el ciclo de entrenamiento con tqdm para visualizar el progreso
        for data in tqdm(train_loader, desc=f"Training Epoch {epoch+1}"):
            user, item, target = data
            optimizer.zero_grad()
            
            if task_type == 'kg' and edge_index is not None and relation_index is not None:
                # Para tareas de KG, asegúrate de pasar edge_index y relation_index
                predictions, _ = model((user, item, target), train_type=task_type, edge_index=edge_index, relation_index=relation_index)
            else:
                # Para tareas de recomendación o cuando no se usan KG
                predictions, _ = model((user, item, target), train_type=task_type)
            
            loss = loss_function(predictions, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Guardar predicciones y objetivos para calcular AUC
            train_predictions.extend(predictions.sigmoid().detach().cpu().numpy())
            train_targets.extend(target.cpu().numpy())
        
        train_loss /= len(train_loader)
        train_auc = roc_auc_score(train_targets, train_predictions)  # Calcular AUC para el entrenamiento
        print(f'Epoch {epoch+1}: Training Loss: {train_loss:.4f}, Training AUC: {train_auc:.4f}')

    # Evaluación después de todas las épocas de entrenamiento
    model.eval()
    val_loss = 0.0
    targets, predictions = [], []
    with torch.no_grad():
        for data in tqdm(val_loader, desc="Validation"):
            user, item, target = data
            
            if task_type == 'kg' and edge_index is not None:
                prediction, _ = model((user, item, target), train_type=task_type, edge_index=edge_index)
            else:
                prediction, _ = model((user, item, target), train_type=task_type)
            
            loss = loss_function(prediction, target)
            val_loss += loss.item()
            predictions.extend(prediction.sigmoid().cpu().numpy())
            targets.extend(target.cpu().numpy())
    
    val_loss /= len(val_loader)
    val_auc = roc_auc_score(targets, predictions)

    print(f'Final Validation Loss: {val_loss:.4f}, Final Validation AUC: {val_auc:.4f}')
    
    return train_loss, val_loss, val_auc






def train_epoch(model, train_loader, optimizer, loss_function, epoch, task_type):
    model.train()
    epoch_loss = 0.0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}")

    for data in progress_bar:
        user, item, target = data
        optimizer.zero_grad()
        if task_type == 'rec':
            predictions, _ = model((user, item, target), train_type='rec')
        else:  # 'kg'
            predictions, true = model((user, item, target), train_type='kg')
        loss = loss_function(predictions, target)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        
        # Actualiza la descripción de la barra de progreso con la pérdida actual
        progress_bar.set_description(f"Epoch {epoch+1} Loss: {epoch_loss / (progress_bar.n + 1)}")
    
    # Imprime la pérdida promedio de la época al finalizar todas las iteraciones
    final_loss = epoch_loss / len(train_loader)
    return final_loss




    

def evaluate_model(model, eval_loader, epoch):
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for data in eval_loader:
            user, item, target = data
            prediction, _ = model((user, item, target), train_type='rec')
            predictions.extend(prediction.view(-1).tolist())
            targets.extend(target.tolist())
    auc = roc_auc_score(targets, predictions)
    print(f"Epoch {epoch+1}: AUC = {auc}")
    return auc



def multi_loss(pred, target, task_type, loss_fn=torch.nn.MSELoss(), kg_loss_fn=None):
    """
    Calcula la pérdida para diferentes tareas.
    
    Parámetros:
    - pred: Predicciones del modelo.
    - target: Objetivos/etiquetas reales.
    - task_type: Tipo de tarea ('rec' para recomendación, 'kg' para grafos de conocimiento).
    - loss_fn: Función de pérdida de PyTorch a utilizar para tareas de recomendación.
    - kg_loss_fn: Función de pérdida de PyTorch a utilizar para tareas de grafos de conocimiento.
    
    Retorna:
    - Valor de pérdida calculado.
    """
    if task_type == "rec":
        return loss_fn(pred, target)
    else:
        # Verifica si se ha proporcionado una función de pérdida específica para KG
        if kg_loss_fn is not None:
            return kg_loss_fn(pred, target)
        else:
            # Retorna un valor de pérdida predeterminado o genera un error si kg_loss_fn no se ha especificado
            raise ValueError("No se ha especificado una función de pérdida para tareas de KG.")






class TrainDatasetWithNegatives(Dataset):
    def __init__(self, positive_triplets, num_entities, num_negative_samples=1):
        self.positive_triplets = positive_triplets
        self.num_entities = num_entities
        self.num_negative_samples = num_negative_samples

    def __len__(self):
        return len(self.positive_triplets)

    def __getitem__(self, idx):
        positive_triplet = self.positive_triplets[idx]
        negative_triplets = []
        for _ in range(self.num_negative_samples):
            negative_tail = np.random.randint(0, self.num_entities)
            while negative_tail == positive_triplet[2]:
                negative_tail = np.random.randint(0, self.num_entities)
            negative_triplets.append((positive_triplet[0], positive_triplet[1], negative_tail))
        
        return positive_triplet, negative_triplets




def kg_loss_with_negative_sampling(pos_pred, neg_pred, margin=1.0):
    # Pérdida de margen
    return torch.mean(torch.clamp(margin - pos_pred + neg_pred, min=0))


def train_and_evaluate_neg(model, rec_train_loader, kg_train_loader, val_loader, optimizer, rec_loss_fn, kg_loss_fn, epoch, task_type, edge_index, relation_index):
    model.train()
    train_loss = 0.0

    # Elegir el DataLoader correcto según el tipo de tarea
    train_loader = kg_train_loader if task_type == 'kg' else rec_train_loader

    for data in train_loader:
        optimizer.zero_grad()

        if task_type == 'rec':
            user, item, target = data
            predictions, _ = model((user, item, target), train_type=task_type)
            loss = rec_loss_fn(predictions, target)
        else:  # 'kg'
            head, relation, tail, label = data
            pos_pred, neg_pred = model(head, relation, tail, task_type=task_type, edge_index=edge_index, relation_index=relation_index)
            loss = kg_loss_fn(pos_pred, neg_pred)

        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f'Epoch {epoch}: Training Loss: {train_loss / len(train_loader)}')

    # Evaluación
    model.eval()
    val_loss = 0.0
    targets, predictions = [], []
    with torch.no_grad():
        for data in val_loader:
            if task_type == 'rec':
                user, item, target = data
                predictions, _ = model((user, item, target), train_type=task_type)
                loss = rec_loss_fn(predictions, target)
            else:  # 'kg'
                # Para simplificar, esta parte se omite en la explicación, pero la lógica sería similar a la del entrenamiento
                pass
            val_loss += loss.item()
            predictions.extend(predictions.view(-1).cpu().numpy())
            targets.extend(target.view(-1).cpu().numpy())
    
    val_loss /= len(val_loader)
    val_auc = roc_auc_score(targets, predictions) if task_type == 'rec' else None

    print(f'Epoch {epoch}: Validation Loss: {val_loss:.4f}')
    if task_type == 'rec':
        print(f'Validation AUC: {val_auc:.4f}')

    return train_loss / len(train_loader), val_loss, val_auc if task_type == 'rec' else None






import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score
import torch
import numpy as np

def train_and_evaluate_rec(model, train_loader, val_loader, optimizer, loss_fn, epochs=5):
    train_losses = []
    val_losses = []
    train_aucs = []  # Lista para almacenar AUC de entrenamiento
    val_aucs = []
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0
        train_predictions_list, train_targets_list = [], []  # Listas para predicciones y objetivos de entrenamiento
        for user, item, target in train_loader:
            optimizer.zero_grad()
            predictions, _ = model((user, item, target), train_type='rec')
            if predictions.dim() == 1:
                target = target.view(-1)
            else:
                target = target.view(-1, 1)
            
            loss = loss_fn(predictions, target)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_predictions_list.extend(predictions.detach().cpu().numpy())
            train_targets_list.extend(target.detach().cpu().numpy())
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Calcular AUC de entrenamiento
        train_predictions_array = np.array(train_predictions_list)
        train_targets_array = np.array(train_targets_list)
        valid_train_indices = ~np.isnan(train_predictions_array)  # Índices de predicciones no-NaN para entrenamiento
        if np.sum(valid_train_indices) > 0:
            train_auc = roc_auc_score(train_targets_array[valid_train_indices], train_predictions_array[valid_train_indices])
            train_aucs.append(train_auc)
        else:
            train_aucs.append(np.nan)  # Añadir NaN si todas las predicciones son NaN

        # Evaluación
        model.eval()
        val_loss = 0
        predictions_list, targets_list = [], []
        for user, item, target in val_loader:
            predictions, _ = model((user, item, target), train_type='rec')
            loss = loss_fn(predictions, target)
            val_loss += loss.item()
            predictions_list.extend(predictions.detach().cpu().numpy())
            targets_list.extend(target.detach().cpu().numpy())
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        # Calcular AUC de validación
        predictions_array = np.array(predictions_list)
        targets_array = np.array(targets_list)
        valid_indices = ~np.isnan(predictions_array)
        if np.sum(valid_indices) > 0:
            val_auc = roc_auc_score(targets_array[valid_indices], predictions_array[valid_indices])
            val_aucs.append(val_auc)
        else:
            val_aucs.append(np.nan)

        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}")

    # Graficar los resultados
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, epochs + 1), train_losses, label='Training Loss')
    plt.plot(range(1, epochs + 1), val_losses, label='Validation Loss')
    plt.title('Loss over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(range(1, epochs + 1), train_aucs, label='Training AUC')
    plt.plot(range(1, epochs + 1), val_aucs, label='Validation AUC')
    plt.title('AUC over epochs')
    plt.xlabel('Epochs')
    plt.ylabel('AUC')
    plt.legend()

    plt.tight_layout()
    plt.show()
