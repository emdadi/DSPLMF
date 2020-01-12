
# -*- coding: utf-8 -*-
"""
Created on 2018
@author: asus
"""
from __future__ import division

import os
from sklearn.model_selection import  RepeatedKFold
from function import load_matrices, apply_threshold
from lmf import PSLMF
from function import apply_threshold_median,score_to_exact_rank

from function import compute_evaluation_criteria_across_wholeMatrix
import numpy as np




data_folder = os.path.join(os.path.pardir, 'Datasets') # All dataset's matrices folder

observation_mat, cell_sim,cell_sim_2,observation_mat_IC50,drugMat =load_matrices(data_folder) 

seed = [80162,45929]

num_repeats=1 # Number of repetition of 10-fold cross validation
model = PSLMF(c=1, K1=20, K2=20, r=95, lambda_p=0.6, lambda_l=0.6, alpha=0.5,beta=0.1, theta=1.3, max_iter=1000)

kf = RepeatedKFold(n_splits=10, n_repeats=num_repeats)
ACC_new, F1_score_new= 0.0, 0.0
REC_new=0.0
Specificity=0.0
MCC_new=0.0

AUC=0.0
precision_new=0.0
result=0.0



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
   
   
   
    prediction = apply_threshold(scores, 0.6)
   
   
    precision_new_o,fold_ACC_new ,fold_F1_score_new,REC_new_output,Specificity_new,MCC_new_output,AUC_new_new_o,fold_loc_f1 = compute_evaluation_criteria_across_wholeMatrix(true_result, prediction,scores)
    
    ACC_new+=fold_ACC_new
    F1_score_new+=fold_F1_score_new
    REC_new+=REC_new_output
    Specificity+=Specificity_new
    MCC_new=MCC_new_output+MCC_new
    #AUC_new=AUC_new_o+AUC_new
    precision_new=precision_new_o+precision_new
    AUC=AUC_new_new_o+AUC
   
    observation_mat_IC50_mat = np.array(observation_mat_IC50)
   
        
    RR=np.array(observation_mat_IC50_mat[test_index])
   
    pred=scores 
   
    
   
    

ACC_new=round(ACC_new/(10*num_repeats),2)
F1_score_new=round(F1_score_new/(10*num_repeats),2)
REC_new=round(REC_new/(10*num_repeats),2)
MCC_new=round(MCC_new/(10*num_repeats),2)
Specificity=round(Specificity/(10*num_repeats),2)

precision_new=round(precision_new/(10*num_repeats),2)
AUC=round(AUC/(10*num_repeats),2)




print('ACC_new', ACC_new)
print('F1_score_new', F1_score_new)
print('REC_new', REC_new)
print('Specificity', Specificity)
print('MCC_new', MCC_new)
print('AUC', AUC)
print('precision_new', precision_new)
