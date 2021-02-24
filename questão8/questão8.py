import numpy as np
from knn import KNN




#abrindo o arquivo
f= open("wine.data","r")
f1=f.readlines()

#variáveis

t1, t2, t3 = 0, 0, 0
count12, count13 = 0, 0
class1, class2, class3 = 0, 0, 0
total1, total2, total3 = 0, 0, 0
dados=np.zeros((178,13))
treinamento_y, teste_y = [], []
treinamento_x, teste_x = [], []
t,m = [], []

#ajeitando os dados

for j in range(178):
	m.append(int(f1[j][0:1:1]))
	for i in range(len(f1[j])-2):
		if f1[j][2+i] != ',':
			count12 += 1
	
			if (2+i) == (len(f1[j])-2):
	
				t.append(count12)
				count12 = 0
				break	
		else:
	
			t.append(count12)
			count12 = 0
	for i in range(13): 
		dados[j][i] = f1[j][2+i+sum(t[0:i:1]):2+i+sum(t[0:i+1:1]):1]
	t.clear()

#função de precisao

def precisao(y_true, y_pred):
    precisao = np.sum(y_true == y_pred) / len(y_true)
    return precisao

#tamanho das classes dos vinhos

for i in range(178):
	if m[i] == 1:
		class1 +=1
	if m[i] == 2:
		class2 +=1
	if m[i] == 3:
		class3 +=1

#divisão dos dados para treino e teste

for i in range(len(m)):
	if m[i]==1.0 and total1< 0.8*class1:
		treinamento_y.append(m[i])
		treinamento_x.append(dados[i])		
		total1 +=1
	elif m[i]==2.0 and total2< 0.8*class2:
		treinamento_y.append(m[i])
		treinamento_x.append(dados[i])	
		total2 +=1
	elif m[i]==3.0 and total3< 0.8*class3:
		treinamento_y.append(m[i])
		treinamento_x.append(dados[i])	
		total3 +=1
	else:
		teste_y.append(m[i])
		teste_x.append(dados[i])	


#utilização do knn
k = 3
clf = KNN(k=k)
clf.fit(treinamento_x, treinamento_y)
predictions = clf.preditor(teste_x)
print("precisão de", precisao(teste_y, predictions))
