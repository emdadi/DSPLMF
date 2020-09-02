# -*- coding: utf-8 -*-
"""
Created on Sun Jul 07 19:57:30 2019

@author: asus
"""


from sklearn.linear_model import SGDClassifier
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
from sklearn.metrics import mean_squared_error
from sklearn.neural_network import MLPRegressor
from sklearn.datasets.california_housing import fetch_california_housing
from sklearn.neural_network import MLPClassifier
from sklearn import svm,tree
def main():
    
   

    
    data = pd.read_csv('train.csv')
    data_test = open('test.csv','r')
    print(data.shape)
    data.head()
    true_value=[]
    nnn=246420   # the number of training data

    
    prediction = open('prediction_tree.txt','w')
    
    rna = data['rna'].values
    cnv = data['cnv'].values
    mut = data['mut'].values
    liu = data['liu'].values
    X = data.iloc[:, 0].values
    Z= data.iloc[:, 1].values 
    W= data.iloc[:, 2].values            
    Y = data.iloc[:, 3].values 
    
    ll=[]
    for i in range (nnn):
        l=[]
        l.append(X[i])
        l.append(Z[i])
        l.append(W[i])
        
        ll.append(l)
    
    ll=np.array(ll)
    
    
    X = np.array(X)
    
    Y = np.array(Y)
    
    X = X.reshape(nnn, 1) 
    Y = Y.reshape(nnn, 1)
        
    
    

    print("Training MLPRegressor...")
     
    est=tree.DecisionTreeClassifier()
    
    est.fit(ll,Y)
    print('Computing partial dependence plots...')
   
    pred=[]
    for line in data_test:
        
        sep=line.split(',')
        sep=sep[0:len(sep)]
        sep[len(sep)-1]=sep[len(sep)-1][:len(sep[len(sep)-1])-1]
        
        rna=sep[0]
        cnv=sep[1]
        mut=sep[2]
       # liu_true=sep[3]
        
        X1 =np.array( [float(rna),float(cnv),float(mut)])
       
        X1 = X1.reshape(1,3)
        p=est.predict(X1)
        p=float(p)
        pred.append(p)
        
        prediction.write(str(p))
        prediction.write('\n')
        
      
    prediction.close() 
    
   
    
    
  
if __name__ == '__main__':
    main()
