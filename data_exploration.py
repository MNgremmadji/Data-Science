# -*- coding: utf-8 -*-
"""
Created on Sun Jan 13 14:54:33 2019

@author: Auxence
"""

import numpy as np
import pandas as pd
#from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

""" 
Nous faisons une analyse description des variables quantitatives et qualitatives.
On charge les données avec la librairie pandas
"""

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
"""
colonnes du dataset = names
"""
dataset.columns = names 

"""
Affichage des 5 premières observations du dataframe

"""
print(dataset.head(5)) 
"""Repartition des observations par classes 
"""
for i in [1,2,3,4,5,6,7]:
    print("classe : %s, nombre d'observations: %s" % (i, len(dataset['Cover_type'][ dataset['Cover_type'] == i]) ) )
""" Variables quantitatives

"""
data_quant = dataset[ ['Elev', 'Aspect', 'Slope', 'Hor_Dis_To_Hydr', \
         'Ver_Dis_To_Hydr', 'Hor_Dis_To_Ro', 'Hilsh_9am', \
         'Hilsh_Noon', 'Hilsh_3pm', 'Hor_Dis_To_Fire_Pts']]
""" 
Variables qualitatives
"""
data_qual =  dataset[['Wild_Area1',\
         'Wild_Area2', 'Wild_Area3', 'Wild_Area4', \
         'Soil_Type1','Soil_Type2','Soil_Type3','Soil_Type4', 'Soil_Type5','Soil_Type6','Soil_Type7',\
         'Soil_Type8','Soil_Type9','Soil_Type10', 'Soil_Type11','Soil_Type12',\
         'Soil_Type13', 'Soil_Type14', 'Soil_Type15','Soil_Type16','Soil_Type17',\
         'Soil_Type18','Soil_Type19','Soil_Type20','Soil_Type21','Soil_Type22','Soil_Type23',\
         'Soil_Type24','Soil_Type25','Soil_Type26','Soil_Type27','Soil_Type28','Soil_Type29',\
         'Soil_Type30','Soil_Type31','Soil_Type32','Soil_Type33','Soil_Type34','Soil_Type35',\
         'Soil_Type36','Soil_Type37','Soil_Type38','Soil_Type39','Soil_Type40']]
"""
Description des variables quantitatives
"""
print(data_quant.describe(include='all')) 


"""
 Histogramme des variables quantitatives
"""

Xprime= data_quant.as_matrix(data_quant.columns[:])
fig = plt.figure(figsize=(15, 15))
for iii in range(data_quant.shape[1]):
    ax = fig.add_subplot(3,4, (iii+1))
    h = ax.hist(Xprime[:,iii], bins=50, color='steelblue',normed=True, edgecolor='none')
    ax.set_title(data_quant.columns[iii], fontsize=10)

"""
Matrice de correlations
"""
correlation = data_quant.corr()
plt.figure(figsize=(10,10))
sns.heatmap(correlation, vmax=1, square=  False,annot=True,cmap='viridis')

plt.title('Matrice de correlation entre les variables quantitives')

"""Diagramme en secteurs des variables qualitatives
"""
"""
 Diagramme en secteur des variables Wilderness_Areai prenant la valeur 1 , i allant de 1 à 4
"""
labels = ['Wild_Area1','Wild_Area2', 'Wild_Area3', 'Wild_Area4']
sizes = [data_qual['Wild_Area1'].value_counts()[1],
   data_qual['Wild_Area2'].value_counts()[1],data_qual['Wild_Area3'].value_counts()[1], data_qual['Wild_Area4'].value_counts()[1]]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()

"""
Diagramme des variables Soil_Typei prenant la valeurr 1 , i = 1 à 6
"""

labels = data_qual.columns[4:10]
sizes= np.zeros(len(labels))
for j in range(len(labels)):
    sizes[j] = data_qual[labels[j]].value_counts()[1]

fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()

"""
Diagramme des variables Soil_Typei , i = 7 à 12
"""

labells = data_qual.columns[10:13]
sizzes= np.zeros(len(labells))
for j in range(len(labells)):
    sizzes[j] = data_qual[labells[j]].value_counts()[1]

fig1, ax1 = plt.subplots()
ax1.pie(sizzes, labels=labells, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()

"""
Diagramme des variables Soil_Typei , i = 37 à 40
"""

labells = data_qual.columns[40::]
sizzes= np.zeros(len(labells))
for j in range(len(labells)):
    sizzes[j] = data_qual[labells[j]].value_counts()[1]

fig1, ax1 = plt.subplots()
ax1.pie(sizzes, labels=labells, autopct='%1.1f%%', shadow=True)
ax1.axis('equal')
plt.show()

""""
On cherche les variables qualitatives constantes afin de les éliminer
"""
for i in range(len(data_qual.columns)):
    if dataset[data_qual.columns[i]].value_counts()[1] == np.shape(dataset)[0] or dataset[data_qual.columns[i]].value_counts()[0] == np.shape(dataset)[0]:
        dataset = dataset.drop(data_qual.columns[i])

"""
Il y a aucune variable qualitative constante
"""

"""
 Selection des variables pertinentes pour la modélisation
 On travaille avec un échantillon des données pour de raison de
 calcul
"""
"""
Après analyse de la matrice des corrélations, 
Vu que corr(Aspect,Hilsh_3pm) = 0.69 > 0.6
Vu que corr(Hor_Dis_To_Hydr,Ver_Dis_To_Hydr) = 0.61 >0.6
Vu que corr(Hilsh_Noon,Hilsh_3pm) = 0.69 >0.6
On se propose d'éliminer 'Aspect', 'Ver_Dis_To_Hydr' et 'Hilsh_3pm'
"""

dataset_new = dataset.drop(['Aspect','Ver_Dis_To_Hydr', 'Hilsh_3pm'] ,1)
"""
Utilisons l'analyse des composantes principales
"""
scaler = StandardScaler()
X=scaler.fit_transform(dataset_new)
pca = PCA()
pca.fit_transform(X)
explained_variance=pca.explained_variance_ratio_
plt.figure(figsize=(6, 4))
plt.bar(range(len(dataset_new.columns)), explained_variance, alpha=0.5, align='center',
label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()

plt.figure(figsize=(14, 13))
plt.axes([.2, .2, .7, .7])
plt.plot(pca.explained_variance_ratio_, linewidth=2)
plt.axis('tight')
plt.xlabel('n_components')
plt.ylabel('explained_variance_ratio_')
"""
En utilisant le critère de coude, on ne retient que 38 composantes
"""

"""
Travaillons avec un échantillon pour des raisons de calcul
"""

X =dataset_new[dataset_new.columns[:-1]].sample(int((np.shape(dataset_new)[0])/10))
Y =dataset_new[dataset_new.columns[-1]].sample(int((np.shape(dataset_new)[0])/10))

"""
On utilise recursive feature elimination avec randomforestclassifier
"""

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.4, random_state=42)
"""
On crée l'objet de RFE( recursive feature elimination) avec randomforestclassification
"""

   
clf_rf_3 = RandomForestClassifier()   

rfe = RFE(estimator=clf_rf_3, n_features_to_select=38, step=1)
rfe = rfe.fit(x_train, y_train)
print('Les 38 meilleurs attributs choisis par rfe:',x_train.columns[rfe.support_])

"""
Explorons le Tree based feature selection avec random forest classification 
pour voir les 38 variables pertinentes
"""
clf_rf_5 = RandomForestClassifier()      
clr_rf_5 = clf_rf_5.fit(x_train,y_train)
importances = clr_rf_5.feature_importances_
std = np.std([tree.feature_importances_ for tree in clf_rf_5.estimators_],
             axis=0)
indices = np.argsort(importances)[::-1]


print("Feature ranking:")

for f in range(x_train.shape[1]):
    print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))

# Plot the feature importances of the forest
plt.figure(figsize=(10,10))
plt.title("Feature importances")
plt.bar(range(x_train.shape[1]), importances[indices],
       color="g", yerr=std[indices], align="center")
plt.xticks(range(x_train.shape[1]), x_train.columns[indices],rotation=90,fontsize = 4)
plt.xlim([-1, x_train.shape[1]])
plt.show()

"""
On travaillera donc avec  les variables explicatives pertinentes suivantes
'Elev', 'Slope', 'Hor_Dis_To_Hydr', 'Hor_Dis_To_Ro', 'Hilsh_9am',
'Hilsh_Noon', 'Hor_Dis_To_Fire_Pts', 'Wild_Area1', 'Wild_Area2',
 'Wild_Area3', 'Wild_Area4', 'Soil_Type1', 'Soil_Type2', 'Soil_Type3',
 'Soil_Type4', 'Soil_Type5', 'Soil_Type6', 'Soil_Type10', 'Soil_Type11',
 'Soil_Type12', 'Soil_Type13', 'Soil_Type16', 'Soil_Type17',
 'Soil_Type19', 'Soil_Type20', 'Soil_Type22', 'Soil_Type23',
  'Soil_Type24', 'Soil_Type26', 'Soil_Type29', 'Soil_Type30',
  'Soil_Type31', 'Soil_Type32', 'Soil_Type33', 'Soil_Type34',
  'Soil_Type35', 'Soil_Type38', 'Soil_Type39', 'Soil_Type40']

"""
