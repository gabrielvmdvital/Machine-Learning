import csv
import random
import math
import operator
from knn import KNN
import numpy as np

c, d = [], []
class0, class1, total1, total0 = 0, 0, 0, 0
treinamento_y, treinamento_x, teste_x, teste_y = [], [], [], []

#abrindo documento para leitura

with open("pulsos.csv",'r') as csv_documento:

#armazenando os dados em uma lista

	c = list(csv.reader(csv_documento))

#armazenando os dados e separando-os para a lista de paridade

for x in range(400):
	for y in range(2001):
		c[x][y] = float(c[x][y])

#última coluna destinada ao bit de paridade

d = [float(c[i][-1]) for i in range(400)]

#armazenando os dados sem a última coluna

for x in range(400):
	for y in range(2000):
		c[x][y] = float(c[x][y])


#contagem da quantidade das classes

for i in range(400):
	if d[i] == 1.0:
		class1 +=1
	else: 
		class0 +=1

#divisão dos dados para treino e teste

for i in range(len(d)):
	if d[i]==1.0 and total1< 0.8*class1:
		treinamento_y.append(d[i])
		treinamento_x.append(c[i])		
		total1 += 1
	elif d[i]==0 and total0< 0.8*class0:
		treinamento_y.append(d[i])
		treinamento_x.append(c[i])		
		total0 +=0
	else:
		teste_y.append(d[i])
		teste_x.append(c[i])	

#transformando as listas em matrizes

treinamento_y = np.asarray(treinamento_y)
treinamento_x =np.asarray(treinamento_x)
teste_x = np.asarray(teste_x)
teste_y = np.asarray(teste_y)

#função de precisao

def precisao(y_true, y_pred):
    precisao = np.sum(y_true == y_pred) / len(y_true)
    return precisao

#utilização do knn

k = 3
clf = KNN(k=k)
clf.fit(treinamento_x, treinamento_y)
predictions = clf.preditor(teste_x)
print("Acurácia de", precisao(teste_y, predictions))
