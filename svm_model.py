# -*- coding: utf-8 -*-
"""
Created on Mon Jan 14 22:29:09 2019

@author: Auxence
"""

from sklearn import neighbors
from sklearn import svm
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import time
from sklearn import metrics


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
## Separation du dataset en train et test

X_train, X_test, y_train, y_test = model_selection.train_test_split(X,Y, shuffle = 'True',\
                                        test_size=0.20) # 20% des données dans le jeu de test)


clf = svm.SVC(kernel='linear') # Linear Kernel

#Train the model using the training sets
t0=time.time()
clf.fit(X_train, y_train)
print("temps d apprentissage",time.time()-t0)
#Predicion
y_pred = clf.predict(X_test)
# Model Precision: what percentage of positive tuples are labeled as such?
print("Precision:",metrics.precision_score(y_test, y_pred))

# Model Recall: what percentage of positive tuples are labelled as such?
print("Recall:",metrics.recall_score(y_test, y_pred))
