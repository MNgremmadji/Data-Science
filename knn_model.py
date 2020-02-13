# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 20:29:02 2019

@author: Auxence
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import random 
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
from sklearn import neighbors, metrics
import time


dataset= pd.read_table('covtype.csv', sep =',',header=1 )
names = ['Elev', 'Aspect', 'Slope', 'Hor_Dis_To_Hydr', \
         'Ver_Dis_To_Hydr', 'Hor_Dis_To_Ro', 'Hilsh_9am', \
         'Hilsh_Noon', 'Hilsh_3pm', 'Hor_Dis_To_Fire_Pts', 'Wild_Area1',\
         'Wild_Area2', 'Wild_Area3', 'Wild_Area4', \
         'Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4', 'Soil_Type5','Soil_Type6','Soil_Type7',\
         'Soil_Type8','Soil_Type9','Soil_Type10', 'Soil_Type11','Soil_Type12',\
         'Soil_Type13', 'Soil_Type14', 'Soil_Type15','Soil_Type16','Soil_Type17',\
         'Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23',\
         'Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29',\
         'Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35',\
         'Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40', \
         'Cover_type']

dataset.columns = names 


dataset_two= dataset[['Elev', 'Slope', 'Hor_Dis_To_Hydr', 'Hor_Dis_To_Ro', 'Hilsh_9am',
'Hilsh_Noon', 'Hor_Dis_To_Fire_Pts', 'Wild_Area1', 'Wild_Area2',
 'Wild_Area3', 'Wild_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type10', 'Soil_Type11',
 'Soil_Type12', 'Soil_Type13', 'Soil_Type16', 'Soil_Type17',
 'Soil_Type19', 'Soil_Type20', 'Soil_Type22', 'Soil_Type23',
  'Soil_Type24', 'Soil_Type26', 'Soil_Type29', 'Soil_Type30',
  'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
  'Soil_Type35', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40','Cover_type'] ]
X = dataset_two.as_matrix(dataset.columns[:-1])
Y = dataset_two.as_matrix([dataset.columns[-1]])
Y = Y.flatten()
X_train, X_test, y_train, y_test = train_test_split(X,Y, shuffle = 'True',test_size=0.3,random_state = 42)
## Standardisation des données pour avoir des données de même echelle 
std_scale = preprocessing.StandardScaler().fit(X_train)
X_train_std = std_scale.transform(X_train)
X_test_std = std_scale.transform(X_test)

kkk= [1,2,3,4,5,6,7,8,9]
result = []
for j in [1,2,3,4,5,6,7,8,9]:
    modele_one = neighbors.KNeighborsClassifier(n_neighbors = j)
    t0=time.time()
    modele_one.fit(X_train,y_train)
    print("temps d apprentissage",time.time()-t0)
    Z= modele_one.predict(X_test)
    score = accuracy_score(Z,y_test)
    print("j","=",j, "," ,"score sur les donnees de test","=",score)
    result.append(score)
    
"""
Le modèle optimal est 3-NN avec un score de  0.959530475491
sur la base de test
avec temps d'apprentissage 8.240982294082642
"""
plt.figure()
plt.xlabel('k')
plt.ylabel('Score')
plt.title('EVolution du score en fonction de k')
plt.plot(kkk,result)
plt.show() 
print("k optimal:","=",3)
