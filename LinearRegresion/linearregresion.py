#Momento de Retroalimentacion Alan Eduardo Aquino Rosas

#Se implemento linear regresion

#En este caso se trabajo sobre un dataset que incluye caracteristicas de hogares de las casas y sus respectivos costos de renta
#En este ejemplo debido a que es un primer acercamiento a la base de datos para no complicar el processo de preprocesamiento de datos se tomaron 3 caracteristicas basicas del dataset

#Las caracteristicas son:

#BHK Numero de Habitaciones,Espacios y cocinas
#Size Tamano  del hogar en feets
#Bathroom Numero de banos
#La variable dependiente que se quire predecir o la "y gorrito" es el precio de renta 

import pandas as pd
#Se importa el dataset
DF=pd.read_csv("E:\Carrera main\Cursos carerra\Carrera 7 Semestre\Ciencia de datos Avanzada y AI\MachineLearningModule\mymlenviroment\Programs\LinearRegresion\House_Rent_Dataset.csv")
print(DF.head())


dx_bhk=DF["BHK"] #Number of Bedrooms, Hall, Kitchen.
dx_size=DF["Size"] #in feet
dx_bath=DF["Bathroom"]# Number of Bathrooms.


dy=DF["Rent"]
#Inicializan los pesos para computar un primer costo
theta=[1,1,1,1]
#Se elije un tamano de paso
alpha=0.0001
#Hipotesis linear con 3 caracteristicas
h1=lambda theta,x1,x2,x3: theta[0]+theta[1]*x1+theta[2]*x2+theta[3]*x3
n=len(dy)
for reps in range(100):#ciclo 
    for i in range(0,n):#Computacion de batch

        acumDelta=[]
        acumDeltaX=[]
        acumDeltaX2=[]
        acumDeltaX3=[]

        #Inicio proceso de  calculo de los gradientes

        acumDelta.append((h1(theta,dx_bhk[i],dx_size[i],dx_bath[i])-dy[i]))
        acumDeltaX.append((h1(theta,dx_bhk[i],dx_size[i],dx_bath[i])-dy[i])*dx_bhk[i])
        acumDeltaX2.append((h1(theta,dx_bhk[i],dx_size[i],dx_bath[i])-dy[i])*dx_size[i])
        acumDeltaX3.append((h1(theta,dx_bhk[i],dx_size[i],dx_bath[i])-dy[i])*dx_bath[i])
    
        
        sJt0=sum(acumDelta)
        sJt1=sum(acumDeltaX)
        sJt2=sum(acumDeltaX2)
        sJt3=sum(acumDeltaX3)
        
        #Finaliza proceso de calculo de gradientes
        #Se actualizan los gradientes
        theta[0]=theta[0]-(alpha/n)*sJt0
        theta[1]=theta[1]-(alpha/n)*sJt1
        theta[2]=theta[2]-(alpha/n)*sJt2
        theta[3]=theta[3]-(alpha/n)*sJt3
        
#Predicion con 3 cuartos,950 feet y 2 baños        
print("Predicion con 3 cuartos,950 feet y 2 baños: ",h1(theta,3,950,2))    
#Predicion con 1 cuarto,450 feet y 1 baño
print("Predicion 1 cuarto,450 feet y 1 baño: ",h1(theta,1,450,1))
#Pesos resultantes
print("Pesos resultantes",theta) 







