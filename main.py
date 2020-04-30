# -*- coding: utf-8 -*-
"""
Created on Fri Feb 01 11:52:23 2019

@author: asus
"""

from __future__ import division

import os
import numpy as np
import pandas as pd
import scipy.stats
import math
from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score, recall_score,roc_auc_score
from sklearn import metrics
from functions import PSLMF 
from sklearn.model_selection import  RepeatedKFold



data_folder = os.path.join(os.path.pardir, 'Datasets') 

seed = [80162,45929]

def apply_threshold(scores, thre):
    
    prediction = np.zeros_like(scores)
    for i in range(len(scores)):
        threshold = max(scores[i]) - ((max(scores[i]) - min(scores[i])) * (thre))
        for j in range(len(scores[i])):
            if (scores[i][j] <= threshold):
                prediction[i][j] = 0
            else: 
                prediction[i][j] = 1
            
    
    return prediction

def apply_threshold(scores, thre):
    
    prediction = np.zeros_like(scores)
    for i in range(len(scores)):
        threshold = max(scores[i]) - ((max(scores[i]) - min(scores[i])) * (thre))
        for j in range(len(scores[i])):
            if (scores[i][j] <= threshold):
                prediction[i][j] = 0
            else: 
                prediction[i][j] = 1
            
    
    return prediction

def apply_threshold_median(scores):
    count=0
    #print(len(scores[0]))  :98
   # print(len(scores))
    prediction = np.zeros_like(scores)
    for i in range(0, len(scores[0])):
        threshold = np.median ((scores[:,i]))
        print(threshold)
        for j in range(0, len(scores)):
            if scores[j][i] < threshold:
                prediction[j][i] = 0
                
            else: 
                prediction[j][i] = 1
                
   # print(count)
    return prediction
    


def score_to_exact_rank(s):
    
    p=np.argsort(s)[::-1]
    
    return np.argsort(p)


def compute_evaluation_criteria_across_wholeMatrix(true_result, prediction,scores):
    pres = np.zeros(len(true_result))
    recs = np.zeros(len(true_result))
    
    ACC_new=0.0
    F1_score_new=0.0
    REC_new=0.0
    Specificity=0.0
    MCC_new=0.0
    
    AUC_new_new=0.0
    precision_new=0.0
    co=0.0
    tp_final=0
    tn_final=0
    fp_final=0
    fn_final=0
    fmeas=[]
    true_auc=[]
    score_auc=[]
    
    
    for i in range(len(true_result[0])):
        
        
        fpr, tpr, thresholds = metrics.roc_curve(true_result[:,i], scores[:,i], pos_label=1)
       
        
        if (math.isnan(metrics.auc(fpr, tpr))!=True):
            co=co+1
            AUC_new_new+= metrics.auc(fpr, tpr)
       
        
        true_auc.append(true_result[:,i])
        score_auc.append(scores[:,i])
                
        returned= confusion_matrix(true_result[:,i], prediction[:,i]).ravel()
        if len(returned) == 4:
            tn, fp, fn, tp = returned
            
        if len(returned) == 1:
           tn= returned
           fp=0
           fn=0
           tp=0
        
        tp_final+=tp
        tn_final+=tn
        fp_final+=fp
        fn_final+=fn
        
   
      
    ACC_new =((tp_final+tn_final)/ (tp_final + fp_final + fn_final+tn_final))
    F1_score_new =((2*tp_final) / ((2*tp_final)+fp_final+ fn_final))
    REC_new=((tp_final)/(tp_final+fn_final))
       
    precision_new=((tp_final)/(tp_final+fp_final))
   
        
    MCC_new=(((tp_final*tn_final)-(fp_final*fn_final))/(np.sqrt((tp_final+fp_final)*(tp_final+fn_final)*(fp_final+tn_final)*(fn_final+tn_final))))
    
  
        
      
    Specificity=((tn_final)/(tn_final+fp_final))
    
    AUC_new_new=AUC_new_new/(co)
    
 
    return round(precision_new,2),round(ACC_new,2),round(F1_score_new,2),round(REC_new,2),round(Specificity,2),round(MCC_new,2),round(AUC_new_new,2)


with open(os.path.join(data_folder, "matrix of 0 and 1.txt"), "r") as raw:
        
    observation_mat = [line.strip("\n").split() for line in raw]
        
with open(os.path.join(data_folder, "similarity matrix of drugs.txt"), "r") as raw:
    raw.next()
        
    drug_matt = [line.strip("\n").split()[1:] for line in raw]
        
     
with open(os.path.join(data_folder, "Liu_similarity_For_drugs.txt"), "r") as raw:
        
    drug_matt_liu = [line.strip("\n").split()[0:] for line in raw]
        
        

with open(os.path.join(data_folder, "Similarity matrix based on gene expression profile.txt"), "r") as raw:
        
    cell_sim_1 = [line.strip("\n").split()[0:] for line in raw]
        
with open(os.path.join(data_folder, "Similarity matrix based on copy number alteration.txt"), "r") as raw:
        
    cell_sim_2 = [line.strip("\n").split()[0:] for line in raw]
        
        
with open(os.path.join(data_folder, "Similarity matrix based on single nucloetid mutation.txt"), "r") as raw:
        
    cell_sim_3 = [line.strip("\n").split()[0:] for line in raw]
        
        
with open(os.path.join(data_folder, "Liu_similarity of IC50 values.txt"), "r") as raw:
        
    cell_sim_4 = [line.strip("\n").split()[0:] for line in raw]
        
        
with open(os.path.join(data_folder, "Matrix of k-nearest neighbr.txt"), "r") as raw:
        
    cell_sim_new = [line.strip("\n").split()[0:] for line in raw]
        
       
with open(os.path.join(data_folder, "matrix of IC50 values.txt"), "r") as raww:
    raww.next()
    print('****************************')
    observation_mat_IC50 = [line.strip("\n").split()[1:] for line in raww]
        
 
observation_mat = np.array(observation_mat, dtype=np.float64)    
cell_sim_1 = np.array(cell_sim_1, dtype=np.float64)      
drug_mat=np.array(drug_matt, dtype=np.float64) 
drug_matt_liu=np.array(drug_matt_liu, dtype=np.float64)
cell_sim_2 = np.array(cell_sim_2, dtype=np.float64)
cell_sim_3 = np.array(cell_sim_3, dtype=np.float64) 
cell_sim_4 = np.array(cell_sim_4, dtype=np.float64) 
cell_sim_new=np.array(cell_sim_new, dtype=np.float64)
    
#train:
cell_sim = ((1* cell_sim_1) + (5*cell_sim_4 )+(1* cell_sim_2)+(1* cell_sim_3)) / (8)  #compute weighted average of similarities
   
cell_sim_2=cell_sim_new
   
drugMat=drug_mat
    
    





REC=0.0
Specificity=0.0
MCC=0.0
AUC_new=0.0
precision_new=0.0
ACC_new=0.0
F1_score_new=0.0


repeats_number=1 

best_ACC_new=0.0
best_F1_score_new=0.0
best_REC=0.0
best_Specificity=0.0
best_MCC=0.0
best_AUC_new=0.0
best_precision_new=0.0

best_parametrs=[]    
inf=open('result.txt', 'w')
inf.write('threshould')
inf.write('\t')
inf.write('K1')
inf.write('\t')
inf.write('K2')
inf.write('\t')
inf.write('r')
inf.write('\t')
inf.write('lambda_p')
inf.write('\t')
inf.write('lambda_l')
inf.write('\t')
inf.write('alpha')
inf.write('\t')
inf.write('theta')
inf.write('\t')
inf.write('ACC_new')
inf.write('\t')
inf.write('F1_score_new')
inf.write('\t')
inf.write('REC')
inf.write('\t')
inf.write('Specificity')
inf.write('\t')
inf.write('MCC')
inf.write('\t')
inf.write('AUC_new')
inf.write('\n')
    

for the in np.arange (0.4,0.8,0.1):
        
    for K1_a in np.arange (4,10,1) :
        
        for K2_a in np.arange(4,10,1):
            for r_a in np.arange(85,100,5):
                for lambda_p_a in np.arange (0.1,1,0.5):
                    for lambda_l_a in np.arange (0.6,1,0.5):
                        for alpha_a in np.arange(0.1,1,0.3):
                            for theta_a in np.arange(1.0,1.5,0.3):
                                
      
                                model = PSLMF(c=1, K1=K1_a, K2=K2_a, r=r_a, lambda_p=lambda_p_a, lambda_l=lambda_l_a, alpha=alpha_a, theta=theta_a, max_iter=100)
                                kf = RepeatedKFold(n_splits=10, n_repeats=repeats_number)
                                ACC_new, F1_score_new= 0.0, 0.0
                                REC=0.0
                                Specificity=0.0
                                MCC=0.0
                                AUC_new=0.0
                                
                                
    
    
                                for train_index, test_index, in kf.split(cell_sim, observation_mat):
                                   
                                        
                                  
                                    test_location_mat = np.array(observation_mat)
                                    
                                    test_location_mat[train_index] = 0
                                    train_location_mat = np.array(observation_mat - test_location_mat)
                                   
                                    true_result = np.array(test_location_mat[test_index])
                                    
                                    x = np.repeat(test_index, len(observation_mat[0]))
                                    y = np.arange(len(observation_mat[0]))
                                    y = np.tile(y, len(test_index))
                                
                                    model.fix_model(train_location_mat, train_location_mat,drugMat, cell_sim,cell_sim_2, seed)
                                    scores = np.reshape(model.predict_scores(zip(x, y)), true_result.shape)
                                    pred_Ic50_values=np.reshape(model.predict_scores_2(zip(x, y)), true_result.shape)
                                   
                                    prediction = apply_threshold(scores, the)
                                    precision_new_o,fold_ACC_new ,fold_F1_score_new,REC_output,Specificity_new,MCC_output,AUC_new_o= compute_evaluation_criteria_across_wholeMatrix(true_result, prediction,scores)
                                    
                                    ACC_new+=fold_ACC_new
                                    F1_score_new+=fold_F1_score_new
                                    REC+=REC_output
                                    Specificity+=Specificity_new
                                    MCC=MCC_output+MCC
                                    AUC_new=AUC_new_o+AUC_new
                                    precision_new=precision_new_o+precision_new
                                    
                                    
                                    
                                   
                                    observation_mat_IC50_mat = np.array(observation_mat_IC50)
                                   
                                    RR=np.array(observation_mat_IC50_mat[test_index])
                                    
                                    pred=scores 
                                   
                                       
                                    
                                
                                ACC_new=round(ACC_new/(10*repeats_number),2)
                                F1_score_new=round(F1_score_new/(10*repeats_number),2)
                                REC=round(REC/(10*repeats_number),2)
                                MCC=round(MCC/(10*repeats_number),2)
                                Specificity=round(Specificity/(10*repeats_number),2)
                                AUC_new=round(AUC_new/(10*repeats_number),2)
                                precision_new=round(precision_new/(10*repeats_number),2)
                               
                                
                               
                                if ((ACC_new+F1_score_new+REC+MCC+Specificity+precision_new)> (best_ACC_new+best_F1_score_new+best_REC+best_MCC+best_Specificity+best_precision_new) ) :
                                    best_ACC_new=ACC_new
                                    best_F1_score_new=F1_score_new
                                    best_REC=REC
                                    best_MCC=MCC
                                    best_precision_new=precision_new
                                    best_Specificity=Specificity
                                    best_AUC_new=AUC_new
                                    best_parametrs=[]
                                    best_parametrs.append(the)
                                    best_parametrs.append(K1_a)
                                    best_parametrs.append(K2_a)
                                    best_parametrs.append(r_a)
                                    best_parametrs.append(lambda_p_a)
                                    best_parametrs.append(lambda_l_a)
                                    best_parametrs.append(alpha_a)
                                    best_parametrs.append(theta_a)
                                    best_parametrs.append(ACC_new)
                                    best_parametrs.append(F1_score_new)
                                    best_parametrs.append(REC)
                                    best_parametrs.append(MCC)
                                    best_parametrs.append(Specificity)
                                    best_parametrs.append(AUC_new)
                                    best_parametrs.append(precision_new)
                                    best_parametrs.append(ACC_new+F1_score_new+REC+MCC+Specificity+precision_new)
                                
                                print('the,K1_a,K2_a,r_a,lambda_p_a,lambda_l_a,alpha_a ,theta_a',the,K1_a,K2_a,r_a,lambda_p_a,lambda_l_a,alpha_a ,theta_a)
                                
                                print('ACC_new', ACC_new)
                                print('F1_score_new', F1_score_new)
                                print('REC', REC)
                                print('Specificity', Specificity)
                                print('MCC', MCC)
                                print('AUC_new', AUC_new)
                                print('precision_new', precision_new)
                                
    
                                print('best_parametrs', best_parametrs)
                                
                                
                                inf.write(str(the))
                                inf.write('\t')
                                inf.write(str(K1_a))
                                inf.write('\t')
                                inf.write(str(K2_a))
                                inf.write('\t')
                                inf.write(str(r_a))
                                inf.write('\t')
                                inf.write(str(lambda_p_a))
                                inf.write('\t')
                                inf.write(str(lambda_l_a))
                                inf.write('\t')
                                inf.write(str(alpha_a))
                                inf.write('\t')
                                inf.write(str(theta_a))
                                inf.write('\t')
                                inf.write(str(ACC_new))
                                inf.write('\t')
                                inf.write(str(F1_score_new))
                                inf.write('\t')
                                inf.write(str(REC))
                                inf.write('\t')
                                inf.write(str(Specificity))
                                inf.write('\t')
                                inf.write(str(MCC))
                                inf.write('\t')
                                inf.write(str(AUC_new))
                                inf.write('\t')
                                inf.write(str(precision_new))
                                inf.write('\t')
                                inf.write(str(ACC_new+F1_score_new+REC+MCC+Specificity+precision_new))
                                inf.write('\n')
                                
                                
print('best_parametrs', best_parametrs)

inf.close()









