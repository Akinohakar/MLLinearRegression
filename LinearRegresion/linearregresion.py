#Momento de Retroalimentacion Alan Eduardo Aquino Rosas

#Se implemento linear regresion

#En este caso se trabajo sobre un dataset respecto a los salarios de Data Science Job Salaries donde se incluyen varias caracterisiticas que influyen al salario de una persona en dolares
#En este caso no se considerara caracteristicas como Company Location,Job title y Employe Residence debido a que si se hace 1-Hot-Encoding,se harian una cantitidad muy grande de caracteristicas(Aprox 140) de cada pais,debido a que se debe hacer manualmente seria un proceso muy lento,se podria ocupar numpy pero esta cuanta como libreria
#Se podria considerar label encoding para que no sean varias columnas de subcarateristicas,sin embargo debido a la naturaleza de estte algoritmo  esto podria llegar establecer una relacion entre paises dependiendo el orden alfabetico, debido a esto y  al objetivo de esta practica se descartaran completamente estas 2 caracteristicas 

#Las caracteristicas son:

#BHK Numero de Habitaciones,Espacios y cocinas
#Size Tamano  del hogar en feets
#Bathroom Numero de banos
#La variable dependiente que se quire predecir o la "y gorrito" es el precio de renta 

import pandas as pd
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

#-----------------COSTANTS-------------------------------
ROUTE='./Dataset/ds_salaries.csv'
REPS=500
debugModel=True
#--------------------------------------------------------

#------------------Preparacion de datos-----------------


DF=pd.read_csv(ROUTE)#Se importa el dataset
print(DF.head())#se checha la lectura correcta del dataset

features=["work_year","experience_level","employment_type","job_title","remote_ratio","company_size"]

DF_X=DF[features]
DF_Y=DF["salary_in_usd"]
print(DF_X.head())
print(DF_Y.head())
df_x_work_year=pd.get_dummies(DF_X["work_year"],prefix="Year")#Correcto
df_x_level=pd.get_dummies(DF_X["experience_level"],prefix="Level")#Correcto
df_x_Type_Emp=pd.get_dummies(DF_X["employment_type"],prefix="Type_Emp")#Correcto
df_x_Remote=pd.get_dummies(DF_X["remote_ratio"],prefix='Remote_%')#Correcto
df_x_Size_comp=pd.get_dummies(DF_X["company_size"],prefix='Size_comp')#Correcto

df_x_clean=DF_X.drop(["work_year","experience_level","employment_type","job_title","remote_ratio","company_size"],axis=1)#Columns
df_x_clean=pd.concat([df_x_work_year,df_x_level,df_x_Type_Emp,df_x_Remote,df_x_Size_comp],axis=1)
print(df_x_clean.tail())


train_x,test_x,train_y,test_y=train_test_split(df_x_clean,DF_Y,random_state=0)

print(train_x.columns)

train_Year_2020=train_x[train_x.columns[0]].tolist()
train_Year_2021=train_x[train_x.columns[1]].tolist()
train_Year_2022=train_x[train_x.columns[2]].tolist()
train_Level_EN=train_x[train_x.columns[3]].tolist()
train_Level_EX=train_x[train_x.columns[4]].tolist()
train_Level_MI=train_x[train_x.columns[5]].tolist()
train_Level_SE=train_x[train_x.columns[6]].tolist()
train_Type_Emp_CT=train_x[train_x.columns[7]].tolist()
train_Type_Emp_FL=train_x[train_x.columns[8]].tolist()
train_Type_Emp_FT=train_x[train_x.columns[9]].tolist()
train_Type_Emp_PT=train_x[train_x.columns[10]].tolist()
train_Remote_0=train_x[train_x.columns[11]].tolist()
train_Remote_50=train_x[train_x.columns[12]].tolist()
train_Remote_100=train_x[train_x.columns[13]].tolist()
train_Size_comp_L=train_x[train_x.columns[14]].tolist()
train_Size_comp_M=train_x[train_x.columns[15]].tolist()
train_Size_comp_S=train_x[train_x.columns[16]].tolist()
train_y=train_y.tolist()





#Standarization




#Inicializan los pesos para computar un primer costo
theta=[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
print(len(theta))


#Se elije un tamano de paso
alpha=0.09


#--------FUNCTIONS------------------------
#Hipotesis linear con 17 caracteristicas
h1=lambda theta,X: theta[0]+theta[1]*X[0]+theta[2]*X[1]+theta[3]*X[2]+theta[4]*X[3]+theta[5]*X[4]+theta[6]*X[5]+theta[7]*X[6]+theta[8]*X[7]+theta[9]*X[8]+theta[10]*X[9]+theta[11]*X[10]+theta[12]*X[11]+theta[13]*X[12]+theta[14]*X[13]+theta[15]*X[14]+theta[16]*X[15]+theta[17]*X[16] 
lossfuction=lambda y_gorrito,y: (y_gorrito-y)**2 #Funcion de pierde
#--------FUNCTIONS------------------------

n=len(train_y)

trainingCost=[]
numberofepochs=[]




for reps in range(REPS):#ciclo 
    
    
    acumDelta=[]
    acumDeltaX=[]
    acumDeltaX2=[]
    acumDeltaX3=[]
    acumDeltaX4=[]
    acumDeltaX5=[]
    acumDeltaX6=[]
    acumDeltaX7=[]
    acumDeltaX8=[]
    acumDeltaX9=[]
    acumDeltaX10=[]
    acumDeltaX11=[]
    acumDeltaX12=[]
    acumDeltaX13=[]
    acumDeltaX14=[]
    acumDeltaX15=[]
    acumDeltaX16=[]
    acumDeltaX17=[]
    traningCostAcumulation=0
    
    for i in range(0,n):#Computacion de batch
        #Inicio proceso de  calculo de los gradientes
        X=[train_Year_2020[i],train_Year_2021[i],train_Year_2022[i],train_Level_EN[i],train_Level_EX[i],train_Level_MI[i],train_Level_SE[i],train_Type_Emp_CT[i],train_Type_Emp_FL[i],train_Type_Emp_FT[i],train_Type_Emp_PT[i],train_Remote_0[i],train_Remote_50[i],train_Remote_100[i],train_Size_comp_L[i],train_Size_comp_M[i],train_Size_comp_S[i]]#Se costruye arreglo de caracteristicas
        
       
        
        acumDelta.append((h1(theta,X)-train_y[i]))
        acumDeltaX.append((h1(theta,X)-train_y[i])*X[0])
        acumDeltaX2.append((h1(theta,X)-train_y[i])*X[1])
        acumDeltaX3.append((h1(theta,X)-train_y[i])*X[2])
        acumDeltaX4.append((h1(theta,X)-train_y[i])*X[3])
        acumDeltaX5.append((h1(theta,X)-train_y[i])*X[4])
        acumDeltaX6.append((h1(theta,X)-train_y[i])*X[5])
        acumDeltaX7.append((h1(theta,X)-train_y[i])*X[6])
        acumDeltaX8.append((h1(theta,X)-train_y[i])*X[7])
        acumDeltaX9.append((h1(theta,X)-train_y[i])*X[8])
        acumDeltaX10.append((h1(theta,X)-train_y[i])*X[9])
        acumDeltaX11.append((h1(theta,X)-train_y[i])*X[10])
        acumDeltaX12.append((h1(theta,X)-train_y[i])*X[11])
        acumDeltaX13.append((h1(theta,X)-train_y[i])*X[12])
        acumDeltaX14.append((h1(theta,X)-train_y[i])*X[13])
        acumDeltaX15.append((h1(theta,X)-train_y[i])*X[14])
        acumDeltaX16.append((h1(theta,X)-train_y[i])*X[15])
        acumDeltaX17.append((h1(theta,X)-train_y[i])*X[16])
       
        #Computacion del costo
        traningCostAcumulation+=lossfuction(h1(theta,X),train_y[i])
        
     
    sJt0=sum(acumDelta)
    sJt1=sum(acumDeltaX)
    sJt2=sum(acumDeltaX2)
    sJt3=sum(acumDeltaX3)
    sJt4=sum(acumDeltaX4)
    sJt5=sum(acumDeltaX5)
    sJt6=sum(acumDeltaX6)
    sJt7=sum(acumDeltaX7)
    sJt8=sum(acumDeltaX8)
    sJt9=sum(acumDeltaX9)
    sJt10=sum(acumDeltaX10)
    sJt11=sum(acumDeltaX11)
    sJt12=sum(acumDeltaX12)
    sJt13=sum(acumDeltaX13)
    sJt14=sum(acumDeltaX14)
    sJt15=sum(acumDeltaX15)
    sJt16=sum(acumDeltaX16)
    sJt17=sum(acumDeltaX17)
    
    #Cost function
    epochTrainingCost=traningCostAcumulation/(2*n)
    
    #Para graficar
    trainingCost.append(epochTrainingCost)
    numberofepochs.append(reps)
    
    
    
    #Finaliza proceso de calculo de gradientes
    #Se actualizan los gradientes
    theta[0]=theta[0]-(alpha/n)*sJt0
    theta[1]=theta[1]-(alpha/n)*sJt1
    theta[2]=theta[2]-(alpha/n)*sJt2
    theta[3]=theta[3]-(alpha/n)*sJt3
    theta[4]=theta[4]-(alpha/n)*sJt4
    theta[5]=theta[5]-(alpha/n)*sJt5
    theta[6]=theta[6]-(alpha/n)*sJt6
    theta[7]=theta[7]-(alpha/n)*sJt7
    theta[8]=theta[8]-(alpha/n)*sJt8
    theta[9]=theta[9]-(alpha/n)*sJt9
    theta[10]=theta[10]-(alpha/n)*sJt10
    theta[11]=theta[11]-(alpha/n)*sJt11
    theta[12]=theta[12]-(alpha/n)*sJt12
    theta[13]=theta[13]-(alpha/n)*sJt13
    theta[14]=theta[14]-(alpha/n)*sJt14
    theta[15]=theta[15]-(alpha/n)*sJt15  
    theta[16]=theta[16]-(alpha/n)*sJt16
    theta[17]=theta[17]-(alpha/n)*sJt17  
    
    if(debugModel):
        print(f"Completado el %{reps/5} costo actual= {epochTrainingCost}")
    
   
    
    




def graphCost(number_epochs,cost_per_epoch):
    plt.title("Cost in training set")
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost') 
    plt.plot(number_epochs,cost_per_epoch)
    plt.show()

def computeCost(x,y,theta):
    print(theta)
    test_Year_2020=train_x[x.columns[0]].tolist()
    test_Year_2021=train_x[x.columns[1]].tolist()
    test_Year_2022=train_x[x.columns[2]].tolist()
    test_Level_EN=train_x[x.columns[3]].tolist()
    test_Level_EX=train_x[x.columns[4]].tolist()
    test_Level_MI=train_x[x.columns[5]].tolist()
    test_Level_SE=train_x[x.columns[6]].tolist()
    test_Type_Emp_CT=train_x[x.columns[7]].tolist()
    test_Type_Emp_FL=train_x[x.columns[8]].tolist()
    test_Type_Emp_FT=train_x[x.columns[9]].tolist()
    test_Type_Emp_PT=train_x[x.columns[10]].tolist()
    test_Remote_0=train_x[x.columns[11]].tolist()
    test_Remote_50=train_x[x.columns[12]].tolist()
    test_Remote_100=train_x[x.columns[13]].tolist()
    test_Size_comp_L=train_x[x.columns[14]].tolist()
    test_Size_comp_M=train_x[x.columns[15]].tolist()
    test_Size_comp_S=train_x[x.columns[16]].tolist()
    test_y=y.tolist()
    
    
    
    n=len(y)
    acumcost=0
    for i in range(0,n):
        X_test=[test_Year_2020[i],test_Year_2021[i],test_Year_2022[i],test_Level_EN[i],test_Level_EX[i],test_Level_MI[i],test_Level_SE[i],test_Type_Emp_CT[i],test_Type_Emp_FL[i],test_Type_Emp_FT[i],test_Type_Emp_PT[i],test_Remote_0[i],test_Remote_50[i],test_Remote_100[i],test_Size_comp_L[i],test_Size_comp_M[i],test_Size_comp_S[i]]
        acumcost+=lossfuction(h1(theta,X_test),test_y[i])
    
    cost=acumcost/(2*n)
    return cost

graphCost(numberofepochs,trainingCost)

print("-----------------End of training-----------------------")
print("-----------------Training results-----------------------")     
print("Pesos resultantes",theta)   
print("El costo resultante es:",trainingCost[len(trainingCost)-1])
print("-----------------Test results-----------------------")
print(f"El costo resultante es: {computeCost(test_x,test_y,theta)}")   









