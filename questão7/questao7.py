import math
import numpy as np
import random
import sys

f= open("iris.data","r")
f1=f.readlines()

t1, t2, t3 = 0, 0, 0
dados=np.zeros((150,4))
dados1=[]
dado=[]
cov11 = []
dados2=[]
cov22 = []
dados3=[]
cov33 = []
auxiliar = np.zeros((4,4))
dado=np.zeros((50,4))
dado1=np.zeros((50,4))
dado2=np.zeros((50,4))


for x in range(len(f1)):
    if x<=49:
        dado[x][0] = float("".join(f1[x][0:3:1]))
        dado[x][1] = float("".join(f1[x][4:7:1]))
        dado[x][2] = float("".join(f1[x][8:11:1]))
        dado[x][3]= float("".join(f1[x][12:15:1]))
        
    elif x >= 50 and x <= 99:
        dado1[x-50][0] = float("".join(f1[x][0:3:1]))
        dado1[x-50][1] = float("".join(f1[x][4:7:1]))
        dado1[x-50][2] = float("".join(f1[x][8:11:1]))
        dado1[x-50][3] = float("".join(f1[x][12:15:1]))
        
    elif x >= 100 and x <= 149:
        dado2[x-100][0] = float("".join(f1[x][0:3:1]))
        dado2[x-100][1] = float("".join(f1[x][4:7:1]))
        dado2[x-100][2] = float("".join(f1[x][8:11:1]))
        dado2[x-100][3] = float("".join(f1[x][12:15:1]))
        
for x in range(len(f1)):
    if x<=149 :
        dados[x][0] = float("".join(f1[x][0:3:1]))
        dados[x][1] = float("".join(f1[x][4:7:1]))
        dados[x][2] = float("".join(f1[x][8:11:1]))
        dados[x][3] = float("".join(f1[x][12:15:1]))

aux1 = dado
aux2 = dado1
aux3 = dado2

v1 = []
des1 = []
v2 = []
des2 = []
v3 = []
des3 = []

#calculando a média

for i in range(4):
	media1=0
	media2=0
	media3=0
	for x in range(50): 
		media1 += ((dado[x][i]/50))
		media2 += ((dado1[x][i]/50))
		media3 += ((dado2[x][i]/50))
	v1.append(round(media1,3))
	v2.append(round(media2,3))
	v3.append(round(media3,3))

print("O vetor de média da primeira classe é: ", v1)
print()
print("O vetor de média da segundaa classe é: ", v2)
print()
print("O vetor de média da terceira classe é: ", v3)
print()

# covariância

for i in range (50):
	aux1[i] = aux1[i] - v1
	aux2[i] = aux2[i] - v2
	aux3[i] = aux3[i] - v3


cov1 = np.matmul(aux1.T,aux1)/50
cov2 = np.matmul(aux2.T,aux2)/50
cov3 = np.matmul(aux3.T,aux3)/50

for i in range(4):

###################################classe 1

	cov1[i][0] = round(cov1[i][0],3)
	cov1[i][1] = round(cov1[i][1],3)
	cov1[i][2] = round(cov1[i][2],3)
	cov1[i][3] = round(cov1[i][3],3)

####################################class 2

	cov2[i][0] = round(cov2[i][0],3)
	cov2[i][1] = round(cov2[i][1],3)
	cov2[i][2] = round(cov2[i][2],3)
	cov2[i][3] = round(cov2[i][3],3)

####################################classe 3

	cov3[i][0] = round(cov3[i][0],3)
	cov3[i][1] = round(cov3[i][1],3)
	cov3[i][2] = round(cov3[i][2],3)
	cov3[i][3] = round(cov3[i][3],3)

#colocando o padrao igual ao scilab [1,2];[1,2]
a1=np.ndarray
a2=np.ndarray
a3=np.ndarray

cov1 = a1.tolist(cov1)
cov2 = a2.tolist(cov2)
cov3 = a3.tolist(cov3)


des1 = [str(cov1[i]) for i in range(4)]
des2 = [str(cov2[i]) for i in range(4)]
des3 = [str(cov3[i]) for i in range(4)]
print()
for i in range(4):
	cov11.append((des1[i][1:len(des1[i])-1]))
cov11 = " ; ".join(cov11)

a1 = ''
a1 = "["+cov11+"]"

for i in range(4):
	cov22.append((des2[i][1:len(des2[i])-1]))
cov22 = " ; ".join(cov22)

a2 = ''
a2 = "["+cov22+"]"

for i in range(4):
	cov33.append((des3[i][1:len(des3[i])-1]))
cov33 = " ; ".join(cov33)

a3 = ''
a3 = "["+cov33+"]"

print("A matriz de covariância da primeira classe é: ")
print(a1)
print()
######
print("A matriz de covariância da segundaa classe é: ")
print(a2)
print()
######
print("A matriz de covariância da terceira classe é: ")
print(a3)
print()
	
#probabilidade a priori

for x in range(len(f1)):
	if x<=149 :
		dados1.append(f1[x][16:30:1])

for i in range(len(dados1)):
	if str(dados1[i]) == 'Iris-virginica':
		t1 += 1
	elif str(dados1[i]) == 'Iris-versicolo':
		t2 += 1
	else:
		t3 += 1
t1 = round(t1/150,3)
t2 = round(t2/150,3)
t3 = round(t3/150,3)
print()
print("A probabilidade da Iris-setosa é:    ", t3)
print("A probabilidade da Iris-virginica é: ", t1)
print("A probabilidade da Iris-versicolo é: ", t2)

