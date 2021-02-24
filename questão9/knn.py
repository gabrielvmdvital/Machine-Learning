import numpy as np
import math
from collections import Counter


def distancia_euclidiana(x, y):
        return np.sqrt(np.sum((x - y)**2))


class KNN:

    def __init__(self, k=3):
        self.k = k

    def fit(self, x1, y1):
        self.treino_x = x1
        self.treino_y = y1

    def preditor(self, x1):
        preditor_saida = [self.predicao(x) for x in x1]
        return np.array(preditor_saida)

    def predicao(self, x):

	# Calcula as distâncias entre x e todos os exemplos no conjunto de treinamento   
        
        distancias = [distancia_euclidiana(x, i) for i in self.treino_x]
        
	# Classifica por distância e retorna os índices dos primeiros k vizinhos

        k1 = np.argsort(distancias)[:self.k]
        
	# Extrai os rótulos das k amostras de treinamento vizinho mais próximo

        k2 = [self.treino_y[i] for i in k1]
  
        # retorna o rótulo de classe mais comum

        aux = Counter(k2).most_common(1)
        return aux[0][0]
