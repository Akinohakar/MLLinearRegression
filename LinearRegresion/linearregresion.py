#Momento de Retroalimentacion Alan Eduardo Aquino Rosas




#-----------------LIBRARIES-------------------------------
import pandas as pd #For dataset reading
from sklearn.model_selection import train_test_split #For Training/Cross-Validation/Test
import matplotlib.pyplot as plt #For graphing
#--------------------------------------------------------

#-----------------PREPROCESING------------------------------
DF=pd.read_csv("./Dataset/auto-mpg.csv")


features=["cylinders","displacement","horsepower","weight","acceleration"]

DF_X=DF[features]
DF_Y=DF["mpg"]
print(DF_X.info())


train_x,test_x,train_y,test_y=train_test_split(DF_X,DF_Y,random_state=0)

train_cylinders=train_x[train_x.columns[0]].tolist()
train_displacement=train_x[train_x.columns[1]].tolist()
train_hoursepower=train_x[train_x.columns[2]].tolist()
train_weight=train_x[train_x.columns[3]].tolist()
train_acceleration=train_x[train_x.columns[4]].tolist()
train_y=train_y.tolist()



#-------------------END PREPROCESING-------------------------

#-------------------HYPERPARAMETERS-----------------------------
#Inicializan los pesos para computar un primer costo
theta=[1,1,1,1,1,1]
alpha=0.00000001
EPOCHS=200
#-------------------------------------------------------------

#---------------------FUNCTIONS-----------------------------------
#Hipotesis linear con 3 caracteristicas
h1=lambda theta,x1,x2,x3,x4,x5: theta[0]+theta[1]*x1+theta[2]*x2+theta[3]*x3+theta[4]*x4+theta[5]*x5
loss=lambda y_gorrito,y: (y_gorrito-y)**2
#------------------------------------------------------------------------

#---------------------CONTROL VARIABLES---------------------------------------
debugModel=True
#-------------------------------------------------------------------------



#--------------------UTILS----------------------------------
def graphCost(x,y):
    plt.title("Cost in training set")
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost') 
    plt.plot(x,y)
    plt.show()

def computeCost(x,y,t):
    test_cylinders=x[x.columns[0]].tolist()
    test_displacement=x[x.columns[1]].tolist()
    test_hoursepower=x[x.columns[2]].tolist()
    test_weight=x[x.columns[3]].tolist()
    test_acceleration=x[x.columns[4]].tolist()
    y=y.tolist()
    acumLoss=0
    for i in range(0,len(y)):
        acumLoss+=loss(h1(t,test_cylinders[i],test_displacement[i],test_hoursepower[i],test_weight[i],test_acceleration[i]),y[i])
    cost=acumLoss/(2*len(y))
    return cost
    
    
n=len(train_y)

cost_per_iter=[]
actual_eps=[]
for reps in range(EPOCHS):#ciclo 
    acumDelta=[]
    acumDeltaX=[]
    acumDeltaX2=[]
    acumDeltaX3=[]
    acumDeltaX4=[]
    acumDeltaX5=[]
    cost_each_example=0
    for i in range(0,n):#Computacion de batch

        #Inicio proceso de  calculo de los gradientes

        acumDelta.append((h1(theta,train_cylinders[i],train_displacement[i],train_hoursepower[i],train_weight[i],train_acceleration[i])-train_y[i]))
        acumDeltaX.append((h1(theta,train_cylinders[i],train_displacement[i],train_hoursepower[i],train_weight[i],train_acceleration[i])-train_y[i])*train_cylinders[i])
        acumDeltaX2.append((h1(theta,train_cylinders[i],train_displacement[i],train_hoursepower[i],train_weight[i],train_acceleration[i])-train_y[i])*train_displacement[i])
        acumDeltaX3.append((h1(theta,train_cylinders[i],train_displacement[i],train_hoursepower[i],train_weight[i],train_acceleration[i])-train_y[i])*train_hoursepower[i])
        acumDeltaX4.append((h1(theta,train_cylinders[i],train_displacement[i],train_hoursepower[i],train_weight[i],train_acceleration[i])-train_y[i])*train_weight[i])
        acumDeltaX5.append((h1(theta,train_cylinders[i],train_displacement[i],train_hoursepower[i],train_weight[i],train_acceleration[i])-train_y[i])*train_acceleration[i])
        
        #Computing Cost
        cost_each_example+=loss(h1(theta,train_cylinders[i],train_displacement[i],train_hoursepower[i],train_weight[i],train_acceleration[i]),train_y[i])
    
    #Storing cost 
    cost_per_iter.append(cost_each_example/(2*n))
    actual_eps.append(reps)
    sJt0=sum(acumDelta)
    sJt1=sum(acumDeltaX)
    sJt2=sum(acumDeltaX2)
    sJt3=sum(acumDeltaX3)
    sJt4=sum(acumDeltaX4)
    sJt5=sum(acumDeltaX5)
        
    #Finaliza proceso de calculo de gradientes
    #Se actualizan los gradientes
    theta[0]=theta[0]-(alpha/n)*sJt0
    theta[1]=theta[1]-(alpha/n)*sJt1
    theta[2]=theta[2]-(alpha/n)*sJt2
    theta[3]=theta[3]-(alpha/n)*sJt3
    theta[4]=theta[4]-(alpha/n)*sJt4
    theta[5]=theta[5]-(alpha/n)*sJt5
    
    
    if(debugModel):
        print(f"Completado el %{reps/2} costo actual = {cost_each_example/(2*n)}")
        
print("-------------------------End of training------------------------")
print("-------------------------Training results------------------------")        
print(theta)        
print("El costo resultante es: ",cost_per_iter[len(cost_per_iter)-1])
print("----------------------Test results-----------------------------")
print(f"El costo resultante es:{computeCost(test_x,test_y,theta)}")
print("Graficando costos a traves del aprendizaje")
graphCost(actual_eps,cost_per_iter)
print("------------------------Predicciones------------------------")
print("Valor de prediccion de millas por galon",h1(theta,8,307,130,3504,12),"valor real",15)



