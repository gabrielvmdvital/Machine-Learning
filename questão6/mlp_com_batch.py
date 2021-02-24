import numpy as np
import random 
import math

# Leitura do Arquivo Iris.data

f= open("iris.data","r")
f1=f.readlines()

# variáveis 

dados=np.zeros((150,4))
p1=np.zeros(150)
p2=np.zeros(150)
p3=np.zeros(150)
p=np.zeros(150)
dado=np.zeros((50,4))
dado1=np.zeros((50,4))
dado2=np.zeros((50,4))
aux, aux1, aux2 = 0,0,0

##################################################
###### Definindo os dados com suas classes #######
##################################################

for x in range(len(f1)):
    if x<=149 :
        dados[x][0] = float("".join(f1[x][0:3:1]))
        dados[x][1] = float("".join(f1[x][4:7:1]))
        dados[x][2] = float("".join(f1[x][8:11:1]))
        dados[x][3] = float("".join(f1[x][12:15:1]))

for x in range(len(f1)):
    if x<=49:
        dado[x][0] = float("".join(f1[x][0:3:1]))
        dado[x][1] = float("".join(f1[x][4:7:1]))
        dado[x][2] = float("".join(f1[x][8:11:1]))
        dado[x][3]= float("".join(f1[x][12:15:1]))
        p1[x] = 1
    elif x >= 50 and x <= 99:
        dado1[x-50][0] = float("".join(f1[x][0:3:1]))
        dado1[x-50][1] = float("".join(f1[x][4:7:1]))
        dado1[x-50][2] = float("".join(f1[x][8:11:1]))
        dado1[x-50][3] = float("".join(f1[x][12:15:1]))
        p2[x]= 1
    elif x >= 100 and x <= 149:
        dado2[x-100][0] = float("".join(f1[x][0:3:1]))
        dado2[x-100][1] = float("".join(f1[x][4:7:1]))
        dado2[x-100][2] = float("".join(f1[x][8:11:1]))
        dado2[x-100][3] = float("".join(f1[x][12:15:1]))
        p3[x]= 1

##################################################
###### Funções de ativação e suas derivadas ######
##################################################

def relud(xi):

        return np.where(xi <= 0, 0, 1)

def relu(xi):
    return np.greater(xi, 0).astype(int)

def sg(xi):
    for y in range(len(xi)):
        try:
            (1 / (1 + (math.exp(-xi[y]))))
        except:
            xi[y]=0
    return (np.array(xi))

def sgd(x):
    return sg(x)*(1-sg(x))

##################################################
################ Função Maior ####################
##################################################

def maior(x,y,z):
    max = x

    if y > max:
        max = y
    if z > max:
        max = z

    return max

##################################################
############## Função de treino ##################
##################################################

def treino(v,ep,ta,p):
    w=np.random.randn(15,4)/1000
    b=np.zeros((15,1))
    w2 = np.random.randn(1,15)/1000
    b2 = 0
    a=np.zeros((15,1))
    e = 0 
    c=0
    tot_dw2 = 0
    tot_dw1 = 0
    tot_db2 = 0
    tot_db1 = 0
    while c<ep :
        c += 1
        for i in random.sample(range(len(v)), len(v)):
            for j in range(5):

   ##################################################
   ################# Feed_Forward ###################
   ##################################################

                x=v[i].reshape((4,1))
                z1 = np.matmul(w,x) + b
                a1 = relu(z1)
                z2 = np.matmul(w2,a1)+b2
                a2 = sg(z2)

   ##################################################
   ############### Back_Propagation #################
   ##################################################

                dz2 = a2 - p[i] 
                dw2=np.matmul(dz2,a1.T)
                db2=dz2
                dz1=np.matmul(w2.T,dz2)*relud(z1)
                dw1=np.matmul(dz1,x.T)
                db1=dz1
                tot_dw2 += dw2
                tot_dw1 += dw1
                tot_db2 += db2
                tot_db1 += db1

   # Tamanho do Batch_size = 5
  
                if j == 4:
                    dw2 = tot_dw2/5
                    dw1 = tot_dw1/5
                    db2 = tot_db2/5
                    db1 = tot_db1/5
                    tot_dw2 = 0
                    tot_dw1 = 0
                    tot_db2 = 0
                    tot_db1 = 0

   # Atualização dos pesos e bias após o batch_size

            w2-=ta*dw2
            w-=ta*dw1
            b-=ta*db1
            b2-=ta*db2

    return w,b,w2,b2

##################################################
########## Definindos os pesos e Bias ############
##################################################

w11,b11,w12,b12=treino(dados,1000,0.001,p1)
w21,b21,w22,b22=treino(dados,1000,0.001,p2)
w31,b31,w32,b32=treino(dados,1000,0.001,p3)

##################################################
############### Função de teste ##################
##################################################
def teste(x,w,b,w2,b2):
    x = x.reshape((4, 1))
    z1 = np.matmul(w, x) + b
    a1 = relu(z1)
    z2 = np.matmul(w2, a1) + b2
    a2 = sg(z2)
    return a2

#y1=teste(X_test[0],w11,b11,w12,b12)
#y2=teste(X_test[0],w21,b21,w22,b22)
#y3=teste(X_test[0],w31,b31,w32,b32)
#print("dados1")
#print(y1,y2,y3)

#y1=teste(X11_test[0],w11,b11,w12,b12)
#y2=teste(X11_test[0],w21,b21,w22,b22)
#y3=teste(X11_test[0],w31,b31,w32,b32)
#print("dados2")
#print(y1,y2,y3)
#y1=teste(X22_test[0],w11,b11,w12,b12)
#y2=teste(X22_test[0],w21,b21,w22,b22)
#y3=teste(X22_test[0],w31,b31,w32,b32)
#print("dados3")
#print(y1,y2,y3)

##################################################
################## Acurácia ######################
##################################################

for i in range(50):
    y1=teste(dado[i],w11,b11,w12,b12)
    y2=teste(dado[i],w21,b21,w22,b22)
    y3=teste(dado[i],w31,b31,w32,b32)
    
    if maior(y1,y2,y3) == y1:
        aux +=1

    x1=teste(dado1[i],w11,b11,w12,b12)
    x2=teste(dado1[i],w21,b21,w22,b22)
    x3=teste(dado1[i],w31,b31,w32,b32)

    if maior(x1,x2,x3) == x2:
        aux1 +=1

    w1=teste(dado2[i],w11,b11,w12,b12)
    w2=teste(dado2[i],w21,b21,w22,b22)
    w3=teste(dado2[i],w31,b31,w32,b32)

    if maior(w1,w2,w3) == w3:
        aux2 +=1

print("Acurácia de ", (aux1+aux+aux2)/150)	
