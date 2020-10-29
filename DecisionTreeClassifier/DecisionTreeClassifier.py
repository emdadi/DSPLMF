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

    prediction = open('prediction.txt','w')


    
   
    data = pd.read_csv('train_1_GDSC.csv')
    data_test = open('test_1_GDSC.csv','r')
    #print(data.shape)
    data.head()
    true_value=[]
      
    
    rna = data['rna'].values
    cnv = data['cnv'].values
    mut = data['mut'].values
    liu = data['liu'].values
    X = data.iloc[:, 0].values
    Z= data.iloc[:, 1].values 
    W= data.iloc[:, 2].values            
    Y = data.iloc[:, 3].values 
    
    ll=[]
    for i in range (246420):
        l=[]
        l.append(X[i])
        l.append(Z[i])
        l.append(W[i])
        
        ll.append(l)
    
    ll=np.array(ll)
    
    
    X = np.array(X)
    
    Y = np.array(Y)
    
    X = X.reshape(246420, 1) 
    Y = Y.reshape(246420, 1)
        
    #print("Training MLPRegressor...")
    est=tree.DecisionTreeClassifier()
    
  
    est.fit(ll,Y)
    
   
    pred=[]
    for line in data_test:
        
        sep=line.split(',')
        sep=sep[0:len(sep)]
        sep[len(sep)-1]=sep[len(sep)-1][:len(sep[len(sep)-1])-1]
        
        rna=sep[0]
        cnv=sep[1]
        mut=sep[2]
        liu_true=sep[3]
        
        X1 =np.array( [float(rna),float(cnv),float(mut)])
       
        X1 = X1.reshape(1,3)
        p=est.predict(X1)
        p=float(p)
        pred.append(p)
        
        true_value.append(float(liu_true))
        prediction.write(str(p))
        prediction.write('\n')
        
      
  
    
    data = pd.read_csv('train_2_GDSC.csv')
    data_test = open('test_2_GDSC.csv','r')
    #print(data.shape)
    data.head()
    true_value=[]
      
    
    rna = data['rna'].values
    cnv = data['cnv'].values
    mut = data['mut'].values
    liu = data['liu'].values
    X = data.iloc[:, 0].values
    Z= data.iloc[:, 1].values 
    W= data.iloc[:, 2].values            
    Y = data.iloc[:, 3].values 
    
    ll=[]
    for i in range (246420):
        l=[]
        l.append(X[i])
        l.append(Z[i])
        l.append(W[i])
        
        ll.append(l)
    
    ll=np.array(ll)
    
    
    X = np.array(X)
    
    Y = np.array(Y)
    
    X = X.reshape(246420, 1) 
    Y = Y.reshape(246420, 1)
        
    #print("Training MLPRegressor...")
    est=tree.DecisionTreeClassifier()
    
  
    est.fit(ll,Y)
    
   
    pred=[]
    for line in data_test:
        
        sep=line.split(',')
        sep=sep[0:len(sep)]
        sep[len(sep)-1]=sep[len(sep)-1][:len(sep[len(sep)-1])-1]
        
        rna=sep[0]
        cnv=sep[1]
        mut=sep[2]
        liu_true=sep[3]
        
        X1 =np.array( [float(rna),float(cnv),float(mut)])
       
        X1 = X1.reshape(1,3)
        p=est.predict(X1)
        p=float(p)
        pred.append(p)
        
        true_value.append(float(liu_true))
        prediction.write(str(p))
        prediction.write('\n')


    data = pd.read_csv('train_3_GDSC.csv')
    data_test = open('test_3_GDSC.csv','r')
    #print(data.shape)
    data.head()
    true_value=[]
      
    
    rna = data['rna'].values
    cnv = data['cnv'].values
    mut = data['mut'].values
    liu = data['liu'].values
    X = data.iloc[:, 0].values
    Z= data.iloc[:, 1].values 
    W= data.iloc[:, 2].values            
    Y = data.iloc[:, 3].values 
    
    ll=[]
    for i in range (246420):
        l=[]
        l.append(X[i])
        l.append(Z[i])
        l.append(W[i])
        
        ll.append(l)
    
    ll=np.array(ll)
    
    
    X = np.array(X)
    
    Y = np.array(Y)
    
    X = X.reshape(246420, 1) 
    Y = Y.reshape(246420, 1)
        
    #print("Training MLPRegressor...")
    est=tree.DecisionTreeClassifier()
    
  
    est.fit(ll,Y)
    
   
    pred=[]
    for line in data_test:
        
        sep=line.split(',')
        sep=sep[0:len(sep)]
        sep[len(sep)-1]=sep[len(sep)-1][:len(sep[len(sep)-1])-1]
        
        rna=sep[0]
        cnv=sep[1]
        mut=sep[2]
        liu_true=sep[3]
        
        X1 =np.array( [float(rna),float(cnv),float(mut)])
       
        X1 = X1.reshape(1,3)
        p=est.predict(X1)
        p=float(p)
        pred.append(p)
        
        true_value.append(float(liu_true))
        prediction.write(str(p))
        prediction.write('\n')


    data = pd.read_csv('train_4_GDSC.csv')
    data_test = open('test_4_GDSC.csv','r')
    #print(data.shape)
    data.head()
    true_value=[]
      
    
    rna = data['rna'].values
    cnv = data['cnv'].values
    mut = data['mut'].values
    liu = data['liu'].values
    X = data.iloc[:, 0].values
    Z= data.iloc[:, 1].values 
    W= data.iloc[:, 2].values            
    Y = data.iloc[:, 3].values 
    
    ll=[]
    for i in range (246420):
        l=[]
        l.append(X[i])
        l.append(Z[i])
        l.append(W[i])
        
        ll.append(l)
    
    ll=np.array(ll)
    
    
    X = np.array(X)
    
    Y = np.array(Y)
    
    X = X.reshape(246420, 1) 
    Y = Y.reshape(246420, 1)
        
    #print("Training MLPRegressor...")
    est=tree.DecisionTreeClassifier()
    
  
    est.fit(ll,Y)
    
   
    pred=[]
    for line in data_test:
        
        sep=line.split(',')
        sep=sep[0:len(sep)]
        sep[len(sep)-1]=sep[len(sep)-1][:len(sep[len(sep)-1])-1]
        
        rna=sep[0]
        cnv=sep[1]
        mut=sep[2]
        liu_true=sep[3]
        
        X1 =np.array( [float(rna),float(cnv),float(mut)])
       
        X1 = X1.reshape(1,3)
        p=est.predict(X1)
        p=float(p)
        pred.append(p)
        
        true_value.append(float(liu_true))
        prediction.write(str(p))
        prediction.write('\n')


        

    data = pd.read_csv('train_5_GDSC.csv')
    data_test = open('test_5_GDSC.csv','r')
    #print(data.shape)
    data.head()
    true_value=[]
      
    
    rna = data['rna'].values
    cnv = data['cnv'].values
    mut = data['mut'].values
    liu = data['liu'].values
    X = data.iloc[:, 0].values
    Z= data.iloc[:, 1].values 
    W= data.iloc[:, 2].values            
    Y = data.iloc[:, 3].values 
    
    ll=[]
    for i in range (246420):
        l=[]
        l.append(X[i])
        l.append(Z[i])
        l.append(W[i])
        
        ll.append(l)
    
    ll=np.array(ll)
    
    
    X = np.array(X)
    
    Y = np.array(Y)
    
    X = X.reshape(246420, 1) 
    Y = Y.reshape(246420, 1)
        
    #print("Training MLPRegressor...")
    est=tree.DecisionTreeClassifier()
    
  
    est.fit(ll,Y)
    
   
    pred=[]
    for line in data_test:
        
        sep=line.split(',')
        sep=sep[0:len(sep)]
        sep[len(sep)-1]=sep[len(sep)-1][:len(sep[len(sep)-1])-1]
        
        rna=sep[0]
        cnv=sep[1]
        mut=sep[2]
        liu_true=sep[3]
        
        X1 =np.array( [float(rna),float(cnv),float(mut)])
       
        X1 = X1.reshape(1,3)
        p=est.predict(X1)
        p=float(p)
        pred.append(p)
        
        true_value.append(float(liu_true))
        prediction.write(str(p))
        prediction.write('\n')


        
   
    prediction.close()





    DR=open('template_matrix.txt','r')
    f1=open('prediction.txt', 'r')

    matrix1 = np.genfromtxt('template_matrix.txt',delimiter='\t')

    t=[]
    for line in f1:
        s=line
        t.append(line)


    tt=np.array(t)
    ttt=len(tt)
    dim=(len(matrix1))
    r=0
    for k in range(dim):
    
        for m in range (dim):
            
            matrix1[k][m]=tt[r]
            r=r+1
        
    for k in range(dim):
    
        for m in range (dim):
            
            if(matrix1[k][m]==0.0):
                matrix1[k][m]=0.01
                
    
        

    np.savetxt('Matrix of k-nearest neighbr.txt', matrix1, delimiter='\t', newline='\n', fmt='%f')
   
    DR.close()
    f1.close()

if __name__ == '__main__':
    main()
